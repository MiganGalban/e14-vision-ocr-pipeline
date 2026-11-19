import csv
from datetime import datetime
import os
from pathlib import Path
import re
import time
import traceback
from typing import Any

from dotenv import load_dotenv
import numpy as np
from supabase import Client, create_client

from src.processing.ocr import (
    extract_e14_ocr_data,
    load_easyocr_reader,
    load_trocr_resources,
)
from src.processing.tools import (
    clean_float,
    consolidate_digits,
    normalize_code,
    process_e14_form,
    scan_e14_pdfs,
)

# Umbrales calibrados sobre formularios reales E-14 2026
MIN_BARCODE_CONFIDENCE = 0.75
MIN_CODE_CONFIDENCE = 0.55
MIN_CELL_NUMERIC_CONFIDENCE = 0.75
MIN_AVG_VOTES_CONFIDENCE = 0.80
NON_DIGIT_SYMBOLS = {"*", "#", "-", "/", "\\", "."}

# Patrones estructurales oficiales del formulario E-14 Segunda Vuelta
BARCODE_PATTERN = re.compile(r"^\d{15}$")
FORM_CODE_PATTERN = re.compile(r"^\d{7}$")


def init_processing_error_log(log_path: Path) -> None:
    """Crea el archivo CSV de auditoría de errores de la Fase 2 si no existe.

    Args:
        log_path (Path): Ruta destino del archivo CSV de errores.
    """
    if not log_path.exists():
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, mode="w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(["archivo_origen", "bandera_error", "detalle", "timestamp"])


def log_processing_error(log_path: Path, source_file: str, flag_name: str, detail: str) -> None:
    """Registra una bandera activa o excepción de procesamiento en el CSV de auditoría.

    Args:
        log_path (Path): Ruta del archivo CSV de auditoría.
        source_file (str): Nombre base del PDF procesado sin extensión.
        flag_name (str): Identificador de la bandera o tipo de error ocurrido.
        detail (str): Descripción diagnóstica exacta del motivo del fallo.
    """
    with open(log_path, mode="a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow([source_file, flag_name, detail, datetime.now().isoformat()])


def preload_stations_map(supabase: Client) -> dict[tuple[str, str, str, str], int]:
    """Descarga la tabla divipole_regis paginada y construye un mapa hash O(1) en memoria.

    Args:
        supabase (Client): Cliente autenticado de Supabase.

    Returns:
        dict[tuple[str, str, str, str], int]: Mapa {(dd, mm, zz, pp): id_puesto}.
    """
    records = []
    batch_size = 1000
    start = 0

    # Extracción paginada excluyendo consulados (dd='88')
    while True:
        end = start + batch_size - 1
        resp = (
            supabase.table("divipole_regis")
            .select("id, dd, mm, zz, pp")
            .neq("dd", "88")
            .range(start, end)
            .execute()
        )
        rows = resp.data
        if not rows:
            break

        records.extend(rows)
        if len(rows) < batch_size:
            break
        start += batch_size

    # Indexación con claves normalizadas para búsqueda en tiempo constante O(1)
    stations_map: dict[tuple[str, str, str, str], int] = {}
    for row in records:
        key = (
            normalize_code(row["dd"], 2),
            normalize_code(row["mm"], 3),
            normalize_code(row["zz"], 2),
            normalize_code(row["pp"], 2),
        )
        stations_map[key] = int(row["id"])

    print(f"[CACHE] Mapa de puestos cargado en memoria: {len(stations_map):,} puestos indexados.")
    return stations_map


def get_pending_stations_map(
    supabase: Client,
    stations_map: dict[tuple[str, str, str, str], int],
) -> tuple[dict[tuple[str, str, str, str], int], set[str]]:
    """Verifica qué actas están completas en todas las tablas hijas antes de omitirlas.

    Args:
        supabase (Client): Cliente autenticado de Supabase.
        stations_map (dict[tuple[str, str, str, str], int]): Catálogo maestro en memoria.

    Returns:
        tuple: (mapa de puestos pendientes, conjunto de archivo_origen completos).
    """
    divipole_records = []
    batch_size = 1000
    start = 0

    # Consulta paginada con Join embebido hacia revision_banderas, votos y auditoria
    while True:
        end = start + batch_size - 1
        resp = (
            supabase.table("divipole")
            .select(
                "puesto_id, archivo_origen, dp_mesa, cod_x, cod_x_2, barras_1, req_revision, "
                "revision_banderas(id), votos(id), auditoria(id)"
            )
            .range(start, end)
            .execute()
        )
        rows = resp.data
        if not rows:
            break

        divipole_records.extend(rows)
        if len(rows) < batch_size:
            break
        start += batch_size

    processed_station_ids: set[int] = set()
    processed_files: set[str] = set()
    corrupt_count = 0

    # Clasificación de completitud: exige tablas hijas o fallo total intencional registrado
    for row in divipole_records:
        has_banderas = bool(row.get("revision_banderas"))
        has_votos = bool(row.get("votos"))
        has_auditoria = bool(row.get("auditoria"))
        is_manual_fail = (
            row.get("req_revision") is True
            and row.get("dp_mesa") is None
            and row.get("cod_x") is None
            and row.get("cod_x_2") is None
            and row.get("barras_1") is None
        )

        if has_banderas and ((has_votos and has_auditoria) or is_manual_fail):
            if row.get("puesto_id") is not None:
                processed_station_ids.add(int(row["puesto_id"]))
            if row.get("archivo_origen") is not None:
                processed_files.add(str(row["archivo_origen"]))
        else:
            corrupt_count += 1

    pending_stations_map = {
        key: pid for key, pid in stations_map.items() if pid not in processed_station_ids
    }

    print("[ESTADO DE COLA]")
    print(f" - Formularios completos en BD     : {len(processed_files):,}")
    if corrupt_count > 0:
        print(f" - Registros a medias detectados   : {corrupt_count:,} (serán reparados)")
    print(f" - Puestos con registros existentes: {len(processed_station_ids):,}")
    print(f" - Puestos totalmente pendientes   : {len(pending_stations_map):,}")

    return pending_stations_map, processed_files


def resolve_station_id(
    stations_map: dict[tuple[str, str, str, str], int],
    ocr_divipole: dict[str, str | None],
) -> int | None:
    """Resuelve el puesto_id en tiempo O(1) a partir de los códigos DIVIPOLE leídos por OCR.

    Args:
        stations_map (dict[tuple[str, str, str, str], int]): Mapa maestro en RAM.
        ocr_divipole (dict[str, str | None]): Diccionario DIVIPOLE extraído de la Página 1.

    Returns:
        int | None: Llave primaria de `divipole_regis` o None si no existe coincidencia.
    """
    dd = normalize_code(ocr_divipole.get("DP_Dept"), 2)
    mm = normalize_code(ocr_divipole.get("DP_Munp"), 3)
    zz = normalize_code(ocr_divipole.get("DP_Zona"), 2)
    pp = normalize_code(ocr_divipole.get("DP_Psto"), 2)

    return stations_map.get((dd, mm, zz, pp))


def evaluate_review_flags(
    source_file: str,
    ocr_data: dict[str, Any] | None,
    station_id: int | None,
    alignment_failed: bool = False,
    ocr_failed: bool = False,
    consolidated_totals: dict[str, int] | None = None,
    avg_confidence: float = 0.0,
    vote_cells_info: list[tuple[str, str, float]] | None = None,
    aud_bools: dict[str, bool] | None = None,
) -> tuple[dict[str, bool], dict[str, str]]:
    """Evalúa independientemente los criterios de revisión manual y genera su diagnóstico.

    Args:
        source_file (str): Nombre base del archivo PDF (ej. '01_001_02_02_029').
        ocr_data (dict[str, Any] | None): Diccionario consolidado de EasyOCR/TrOCR/OMR.
        station_id (int | None): ID resuelto contra `divipole_regis`.
        alignment_failed (bool, opcional): True si falló la alineación geométrica.
        ocr_failed (bool, opcional): True si falló la inferencia en tiempo de ejecución.
        consolidated_totals (dict[str, int] | None, opcional): Totales enteros de votos.
        avg_confidence (float, opcional): Confianza promedio de las 27 casillas TrOCR.
        vote_cells_info (list[tuple[str, str, float]] | None, opcional): Lista de tuplas
            (nombre_casilla, prediccion, confianza).
        aud_bools (dict[str, bool] | None, opcional): Diccionario booleano OMR de Página 2.

    Returns:
        tuple[dict[str, bool], dict[str, str]]: Tupla con (banderas booleanas, detalle por bandera).
    """
    flags = {
        "err_alineacion": alignment_failed,
        "err_inferencia_ocr": ocr_failed,
        "err_puesto_no_resuelto": False,
        "flag_divipole_mismatch": False,
        "flag_archivo_vs_ocr_diff": False,
        "flag_baja_conf_encabezado": False,
        "flag_baja_conf_votos": False,
        "flag_caracter_anomalo": False,
        "flag_descuadre_aritmetico": False,
        "flag_omr_recuento_ambiguo": False,
        "flag_sin_firmas_jurados": False,
    }
    details: dict[str, str] = {}

    # Si hubo fallo total de alineación o inferencia, retorna sin evaluar el contenido
    if ocr_data is None or alignment_failed or ocr_failed:
        return flags, details

    # 1. Integridad contra catálogo oficial DIVIPOLE
    if station_id is None:
        flags["err_puesto_no_resuelto"] = True
        details["err_puesto_no_resuelto"] = (
            f"DIVIPOLE OCR ({ocr_data.get('DP_Dept')}-{ocr_data.get('DP_Munp')}-"
            f"{ocr_data.get('DP_Zona')}-{ocr_data.get('DP_Psto')}) no existe en divipole_regis"
        )

    # 2. Consistencia de encabezado DIVIPOLE entre Página 1 y Página 2
    if bool(ocr_data.get("manual", False)):
        flags["flag_divipole_mismatch"] = True
        details["flag_divipole_mismatch"] = "Encabezado DIVIPOLE Pag 1 != Pag 2"

    # 3. Validación cruzada entre el nombre del archivo (dd_mm_zz_pp_mesa) y la lectura OCR
    file_parts = source_file.split("_")
    if len(file_parts) >= 5:
        expected_tuple = (
            normalize_code(file_parts[0], 2),
            normalize_code(file_parts[1], 3),
            normalize_code(file_parts[2], 2),
            normalize_code(file_parts[3], 2),
            normalize_code(file_parts[4], 3),
        )
        ocr_tuple = (
            normalize_code(ocr_data.get("DP_Dept"), 2),
            normalize_code(ocr_data.get("DP_Munp"), 3),
            normalize_code(ocr_data.get("DP_Zona"), 2),
            normalize_code(ocr_data.get("DP_Psto"), 2),
            normalize_code(ocr_data.get("DP_Mesa"), 3),
        )
        if expected_tuple != ocr_tuple:
            flags["flag_archivo_vs_ocr_diff"] = True
            details["flag_archivo_vs_ocr_diff"] = f"Archivo={expected_tuple} != OCR={ocr_tuple}"

    # 4. Validación estructural y de confianza en Códigos de Barras (15 dígitos) y Código (7 dígitos)
    s_b1 = clean_float(ocr_data.get("s_barras_1"))
    s_b2 = clean_float(ocr_data.get("s_barras_2"))
    s_cod1 = clean_float(ocr_data.get("s_codigo"))
    s_cod2 = clean_float(ocr_data.get("s_codigo_2"))

    b1_str = str(ocr_data.get("Barras_1") or "").strip().replace(" ", "")
    b2_str = str(ocr_data.get("Barras_2") or "").strip().replace(" ", "")
    c1_str = str(ocr_data.get("Codigo") or "").strip()
    c2_str = str(ocr_data.get("Codigo_2") or "").strip()

    header_reasons = []
    # Comprobación de confianza mínima en barras y código X-#-#-#-# X
    if s_b1 < MIN_BARCODE_CONFIDENCE or s_b2 < MIN_BARCODE_CONFIDENCE:
        header_reasons.append(f"confianza_barras(b1={s_b1}, b2={s_b2})")
    if s_cod1 < MIN_CODE_CONFIDENCE or s_cod2 < MIN_CODE_CONFIDENCE:
        header_reasons.append(f"confianza_codigo(c1={s_cod1}, c2={s_cod2})")
    if "Ilegible" in (b1_str, b2_str, c1_str, c2_str):
        header_reasons.append("lectura_ilegible")

    # Comprobación de estructura del código de barras: 15 dígitos, primeros 11 iguales, pág 01 vs 02
    barcodes_valid = (
        bool(BARCODE_PATTERN.match(b1_str))
        and bool(BARCODE_PATTERN.match(b2_str))
        and b1_str[:11] == b2_str[:11]
        and b1_str[11:13] == "01"
        and b2_str[11:13] == "02"
        and b1_str[13:15] == "02"
        and b2_str[13:15] == "02"
    )
    if not barcodes_valid:
        header_reasons.append(f"patron_barras_invalido('{b1_str}' vs '{b2_str}')")

    # Comprobación de formato (7 dígitos) y coincidencia de Codigo entre Página 1 y Página 2
    codes_valid = (
        bool(FORM_CODE_PATTERN.match(c1_str))
        and bool(FORM_CODE_PATTERN.match(c2_str))
        and c1_str == c2_str
    )
    if not codes_valid:
        header_reasons.append(f"codigo_invalido_o_distinto('{c1_str}' vs '{c2_str}')")

    if header_reasons:
        flags["flag_baja_conf_encabezado"] = True
        details["flag_baja_conf_encabezado"] = "; ".join(header_reasons)

    # 5. Confianza de votos manuscritos (TrOCR): < 0.75 en dígito (0-9) o < 0.80 promedio global
    cells_info = vote_cells_info or []
    low_numeric_cells = [
        f"{key}={pred}({conf:.2f})"
        for key, pred, conf in cells_info
        if pred.isdigit() and conf < MIN_CELL_NUMERIC_CONFIDENCE
    ]
    if (avg_confidence < MIN_AVG_VOTES_CONFIDENCE) or low_numeric_cells:
        flags["flag_baja_conf_votos"] = True
        details["flag_baja_conf_votos"] = (
            f"promedio={avg_confidence:.2f}; celdas_bajas={low_numeric_cells}"
        )

    # 6. Detección de símbolos especiales en casillas de votos (*, #, -, /, \, .)
    anomalous_cells = [
        f"{key}='{pred}'" for key, pred, _ in cells_info if pred in NON_DIGIT_SYMBOLS
    ]
    if anomalous_cells:
        flags["flag_caracter_anomalo"] = True
        details["flag_caracter_anomalo"] = f"simbolos={anomalous_cells}"

    # 7. Consistencia aritmética del acta E-14 (votos por opción y nivelación de urna)
    if consolidated_totals:
        calculated_sum = (
            consolidated_totals["v_a"]
            + consolidated_totals["v_b"]
            + consolidated_totals["v_blan"]
            + consolidated_totals["v_nulo"]
            + consolidated_totals["v_s_mrc"]
        )
        urna_neto = consolidated_totals["t_urna"] - consolidated_totals["t_inc"]
        suma_t = consolidated_totals["suma_t"]
        if calculated_sum != suma_t or urna_neto != suma_t:
            flags["flag_descuadre_aritmetico"] = True
            details["flag_descuadre_aritmetico"] = (
                f"suma_candidatos={calculated_sum}, t_urna-t_inc={urna_neto}, suma_t={suma_t}"
            )

    # 8. Auditoría OMR (Página 2): doble marca en recuento y ausencia de firmas de jurados
    if aud_bools:
        if aud_bools["re_count_si"] and aud_bools["re_count_no"]:
            flags["flag_omr_recuento_ambiguo"] = True
            details["flag_omr_recuento_ambiguo"] = "Re_count_si=True y Re_count_no=True"

        juror_keys = ("firma_j1", "firma_j2", "firma_j3", "firma_j4", "firma_j5", "firma_j6")
        if not any(aud_bools[k] for k in juror_keys):
            flags["flag_sin_firmas_jurados"] = True
            details["flag_sin_firmas_jurados"] = "0 de 6 firmas detectadas (Firma_J1..J6=False)"

    return flags, details


def insert_ocr_result(
    supabase: Client,
    ocr_data: dict[str, Any] | None,
    source_file: str,
    station_id: int | None = None,
    alignment_failed: bool = False,
    ocr_failed: bool = False,
    evidence_crops: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Ejecuta una inserción transaccional atómica (ACID) sin JSON vía RPC en PostgreSQL.

    Args:
        supabase (Client): Cliente autenticado de Supabase.
        ocr_data (dict[str, Any] | None): Datos extraídos o None si ocurrió fallo total.
        source_file (str): Nombre base del archivo PDF (sin extensión).
        station_id (int | None, opcional): Llave foránea hacia `divipole_regis`.
        alignment_failed (bool, opcional): Marca fallo geométrico de alineación.
        ocr_failed (bool, opcional): Marca excepción durante inferencia OCR/HTR/OMR.
        evidence_crops (dict[str, str] | None, opcional): Recortes codificados de respaldo.

    Returns:
        dict[str, Any]: Diccionario con 'status', 'id_divipole', 'flags' y 'flag_details'.
    """
    source_file_clean = source_file[:32]

    # Rama 1: Fallo total de alineación o inferencia (inserta solo cabecera + banderas)
    if ocr_data is None:
        flags, details = evaluate_review_flags(
            source_file=source_file_clean,
            ocr_data=None,
            station_id=station_id,
            alignment_failed=alignment_failed,
            ocr_failed=ocr_failed,
        )
        rpc_params = {
            "p_archivo_origen": source_file_clean,
            "p_puesto_id": station_id,
            "p_dp_mesa": None,
            "p_dp_lugr": None,
            "p_cod_x": None,
            "p_s_cod_x": None,
            "p_cod_x_2": None,
            "p_s_cod_x_2": None,
            "p_barras_1": None,
            "p_s_barras_1": None,
            "p_barras_2": None,
            "p_s_barras_2": None,
            "p_es_fallo_total": True,
            "p_err_alineacion": flags["err_alineacion"],
            "p_err_inferencia_ocr": flags["err_inferencia_ocr"],
            "p_err_puesto_no_resuelto": False,
            "p_flag_divipole_mismatch": False,
            "p_flag_archivo_vs_ocr_diff": False,
            "p_flag_baja_conf_encabezado": False,
            "p_flag_baja_conf_votos": False,
            "p_flag_caracter_anomalo": False,
            "p_flag_descuadre_aritmetico": False,
            "p_flag_omr_recuento_ambiguo": False,
            "p_flag_sin_firmas_jurados": False,
            "p_t_e11": None,
            "p_t_urna": None,
            "p_t_inc": None,
            "p_v_a": None,
            "p_v_b": None,
            "p_v_blan": None,
            "p_v_nulo": None,
            "p_v_s_mrc": None,
            "p_suma_t": None,
            "p_confianza_promedio_ocr": None,
            "p_vp_campos": [],
            "p_vp_indices": [],
            "p_vp_preds": [],
            "p_vp_confs": [],
            "p_vp_crops": [],
            "p_aud_observ_jud": False,
            "p_aud_re_count_si": False,
            "p_aud_re_count_no": False,
            "p_aud_count_req_by": False,
            "p_aud_on_behalf_of": False,
            "p_aud_firma_j1": False,
            "p_aud_firma_j2": False,
            "p_aud_firma_j3": False,
            "p_aud_firma_j4": False,
            "p_aud_firma_j5": False,
            "p_aud_firma_j6": False,
            "p_aud_firma_cc_j1": False,
            "p_aud_firma_cc_j2": False,
            "p_aud_firma_cc_j3": False,
            "p_aud_firma_cc_j4": False,
            "p_aud_firma_cc_j5": False,
            "p_aud_firma_cc_j6": False,
            "p_ap_campos": [],
            "p_ap_valores": [],
        }
        res = supabase.rpc("insert_e14_atomic", rpc_params).execute()
        row_res = res.data[0] if isinstance(res.data, list) else res.data
        return {
            "status": row_res["out_status"],
            "id_divipole": row_res["out_id_divipole"],
            "flags": flags,
            "flag_details": details,
        }

    # Rama 2: Extracción exitosa (prepara arreglos paralelos para votos y auditoría)
    evidence_crops = evidence_crops or {}
    raw_mesa = str(ocr_data.get("DP_Mesa") or "").strip()
    dp_mesa_str = raw_mesa[:3].zfill(3) if raw_mesa else None

    votes_mapping = {
        "t_e11": ("Total_E11_1", "Total_E11_2", "Total_E11_3"),
        "t_urna": ("Total_Urna_1", "Total_Urna_2", "Total_Urna_3"),
        "t_inc": ("Total_Incinerados_1", "Total_Incinerados_2", "Total_Incinerados_3"),
        "v_a": ("Votos_A_1", "Votos_A_2", "Votos_A_3"),
        "v_b": ("Votos_B_1", "Votos_B_2", "Votos_B_3"),
        "v_blan": ("V_Blanco_1", "V_Blanco_2", "V_Blanco_3"),
        "v_nulo": ("V_Nulos_1", "V_Nulos_2", "V_Nulos_3"),
        "v_s_mrc": ("V_Sin_Marcar_1", "V_Sin_Marcar_2", "V_Sin_Marcar_3"),
        "suma_t": ("Suma_Total_1", "Suma_Total_2", "Suma_Total_3"),
    }

    # Consolidación de 3 casillas por fila y construcción de arreglos para votos_precision
    consolidated_totals: dict[str, int] = {}
    vp_campos: list[str] = []
    vp_indices: list[int] = []
    vp_preds: list[str] = []
    vp_confs: list[float] = []
    vp_crops: list[str | None] = []
    vote_cells_info: list[tuple[str, str, float]] = []

    for db_field, (k1, k2, k3) in votes_mapping.items():
        consolidated_totals[db_field] = consolidate_digits(
            str(ocr_data.get(k1, "0")),
            str(ocr_data.get(k2, "0")),
            str(ocr_data.get(k3, "0")),
        )
        for idx, key in enumerate([k1, k2, k3], start=1):
            pred_raw = str(ocr_data.get(key, ""))[:10]
            conf = clean_float(ocr_data.get(f"s_{key}", 0.0))
            vp_campos.append(db_field[:10])
            vp_indices.append(idx)
            vp_preds.append(pred_raw)
            vp_confs.append(conf)
            vp_crops.append(evidence_crops.get(key))
            vote_cells_info.append((key, pred_raw, conf))

    avg_confidence = clean_float(np.mean(vp_confs)) if vp_confs else 0.0

    # Construcción de booleanos y porcentajes OMR para auditoria y auditoria_precision
    audit_mapping = {
        "observ_jud": "Observ_jud",
        "re_count_si": "Re_count_si",
        "re_count_no": "Re_count_no",
        "count_req_by": "Count_req_by",
        "on_behalf_of": "On_behalf_of",
        "firma_j1": "Firma_J1",
        "firma_j2": "Firma_J2",
        "firma_j3": "Firma_J3",
        "firma_j4": "Firma_J4",
        "firma_j5": "Firma_J5",
        "firma_j6": "Firma_J6",
        "firma_cc_j1": "Firma_cc_J1",
        "firma_cc_j2": "Firma_cc_J2",
        "firma_cc_j3": "Firma_cc_J3",
        "firma_cc_j4": "Firma_cc_J4",
        "firma_cc_j5": "Firma_cc_J5",
        "firma_cc_j6": "Firma_cc_J6",
    }

    aud_bools: dict[str, bool] = {}
    ap_campos: list[str] = []
    ap_valores: list[float] = []

    for db_field, dict_key in audit_mapping.items():
        omr_tuple = ocr_data.get(dict_key, (False, 0.0))
        aud_bools[db_field] = bool(omr_tuple[0])
        ap_campos.append(db_field[:20])
        ap_valores.append(clean_float(omr_tuple[1], precision=2))

    # Evaluación final de las banderas de calidad antes de persistir en BD
    flags, details = evaluate_review_flags(
        source_file=source_file_clean,
        ocr_data=ocr_data,
        station_id=station_id,
        alignment_failed=False,
        ocr_failed=False,
        consolidated_totals=consolidated_totals,
        avg_confidence=avg_confidence,
        vote_cells_info=vote_cells_info,
        aud_bools=aud_bools,
    )

    # Invocación atómica de la función PostgreSQL con parámetros escalares y arreglos
    rpc_params = {
        "p_archivo_origen": source_file_clean,
        "p_puesto_id": station_id,
        "p_dp_mesa": dp_mesa_str,
        "p_dp_lugr": str(ocr_data["DP_Lugr"])[:150] if ocr_data.get("DP_Lugr") else None,
        "p_cod_x": str(ocr_data["Codigo"])[:15] if ocr_data.get("Codigo") else None,
        "p_s_cod_x": clean_float(ocr_data.get("s_codigo")),
        "p_cod_x_2": str(ocr_data["Codigo_2"])[:15] if ocr_data.get("Codigo_2") else None,
        "p_s_cod_x_2": clean_float(ocr_data.get("s_codigo_2")),
        "p_barras_1": str(ocr_data["Barras_1"])[:20] if ocr_data.get("Barras_1") else None,
        "p_s_barras_1": clean_float(ocr_data.get("s_barras_1")),
        "p_barras_2": str(ocr_data["Barras_2"])[:20] if ocr_data.get("Barras_2") else None,
        "p_s_barras_2": clean_float(ocr_data.get("s_barras_2")),
        "p_es_fallo_total": False,
        "p_err_alineacion": flags["err_alineacion"],
        "p_err_inferencia_ocr": flags["err_inferencia_ocr"],
        "p_err_puesto_no_resuelto": flags["err_puesto_no_resuelto"],
        "p_flag_divipole_mismatch": flags["flag_divipole_mismatch"],
        "p_flag_archivo_vs_ocr_diff": flags["flag_archivo_vs_ocr_diff"],
        "p_flag_baja_conf_encabezado": flags["flag_baja_conf_encabezado"],
        "p_flag_baja_conf_votos": flags["flag_baja_conf_votos"],
        "p_flag_caracter_anomalo": flags["flag_caracter_anomalo"],
        "p_flag_descuadre_aritmetico": flags["flag_descuadre_aritmetico"],
        "p_flag_omr_recuento_ambiguo": flags["flag_omr_recuento_ambiguo"],
        "p_flag_sin_firmas_jurados": flags["flag_sin_firmas_jurados"],
        "p_t_e11": consolidated_totals["t_e11"],
        "p_t_urna": consolidated_totals["t_urna"],
        "p_t_inc": consolidated_totals["t_inc"],
        "p_v_a": consolidated_totals["v_a"],
        "p_v_b": consolidated_totals["v_b"],
        "p_v_blan": consolidated_totals["v_blan"],
        "p_v_nulo": consolidated_totals["v_nulo"],
        "p_v_s_mrc": consolidated_totals["v_s_mrc"],
        "p_suma_t": consolidated_totals["suma_t"],
        "p_confianza_promedio_ocr": avg_confidence,
        "p_vp_campos": vp_campos,
        "p_vp_indices": vp_indices,
        "p_vp_preds": vp_preds,
        "p_vp_confs": vp_confs,
        "p_vp_crops": vp_crops,
        "p_aud_observ_jud": aud_bools["observ_jud"],
        "p_aud_re_count_si": aud_bools["re_count_si"],
        "p_aud_re_count_no": aud_bools["re_count_no"],
        "p_aud_count_req_by": aud_bools["count_req_by"],
        "p_aud_on_behalf_of": aud_bools["on_behalf_of"],
        "p_aud_firma_j1": aud_bools["firma_j1"],
        "p_aud_firma_j2": aud_bools["firma_j2"],
        "p_aud_firma_j3": aud_bools["firma_j3"],
        "p_aud_firma_j4": aud_bools["firma_j4"],
        "p_aud_firma_j5": aud_bools["firma_j5"],
        "p_aud_firma_j6": aud_bools["firma_j6"],
        "p_aud_firma_cc_j1": aud_bools["firma_cc_j1"],
        "p_aud_firma_cc_j2": aud_bools["firma_cc_j2"],
        "p_aud_firma_cc_j3": aud_bools["firma_cc_j3"],
        "p_aud_firma_cc_j4": aud_bools["firma_cc_j4"],
        "p_aud_firma_cc_j5": aud_bools["firma_cc_j5"],
        "p_aud_firma_cc_j6": aud_bools["firma_cc_j6"],
        "p_ap_campos": ap_campos,
        "p_ap_valores": ap_valores,
    }

    res = supabase.rpc("insert_e14_atomic", rpc_params).execute()
    row_res = res.data[0] if isinstance(res.data, list) else res.data
    status = row_res["out_status"]

    # Registro de colisión de llave única puesto+mesa reportado desde PostgreSQL
    if status == "duplicate_station_flagged":
        flags["err_duplicado_mesa"] = True
        details["err_duplicado_mesa"] = (
            f"Colision uk_puesto_mesa (puesto_id={station_id}, dp_mesa={dp_mesa_str}); guardado dp_mesa=NULL"
        )
    else:
        flags["err_duplicado_mesa"] = False

    return {
        "status": status,
        "id_divipole": row_res["out_id_divipole"],
        "flags": flags,
        "flag_details": details,
    }


def execute_ocr_pipeline(pdf_root_dir: str | Path | None = None) -> dict[str, Any]:
    """Orquesta el pipeline completo de Fase 2: escaneo, alineación, OCR/HTR/OMR y carga ACID.

    Args:
        pdf_root_dir (str | Path | None, opcional): Directorio raíz con los PDFs E-14.
            Si es None, usa la variable de entorno `E14_PDF_DIR` o la carpeta `data/`.

    Raises:
        ValueError: Si faltan las credenciales `SUPABASE_URL` o `SUPABASE_KEY`.

    Returns:
        dict[str, Any]: Diccionario resumen con contadores de métricas y frecuencias por bandera.
    """
    start_time = time.time()
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(dotenv_path=project_root / ".env")

    # 1. Validación de credenciales y resolución de rutas
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise ValueError("Variables SUPABASE_URL o SUPABASE_KEY no definidas en el entorno.")

    if pdf_root_dir is None:
        env_pdf_dir = os.getenv("E14_PDF_DIR")
        pdf_root_dir = Path(env_pdf_dir) if env_pdf_dir else project_root / "data"
    else:
        pdf_root_dir = Path(pdf_root_dir)

    # 2. Inicialización del log CSV y escaneo del directorio de PDFs
    errors_csv_path = project_root / "data" / "errores_procesamiento.csv"
    init_processing_error_log(errors_csv_path)

    df_e14 = scan_e14_pdfs(pdf_root_dir)
    total_files = len(df_e14)
    print(f"[FASE 2] Directorio: {pdf_root_dir}")
    print(f"[FASE 2] Total formularios PDF indexados: {total_files}")

    # Inicialización de contadores de balance y auditoría por bandera
    metrics = {
        "total": total_files,
        "inserted_clean": 0,
        "inserted_with_flags": 0,
        "failed_total": 0,
        "skipped": 0,
        "failed_unhandled": 0,
    }
    flag_counts = {
        "err_alineacion": 0,
        "err_inferencia_ocr": 0,
        "err_puesto_no_resuelto": 0,
        "err_duplicado_mesa": 0,
        "flag_divipole_mismatch": 0,
        "flag_archivo_vs_ocr_diff": 0,
        "flag_baja_conf_encabezado": 0,
        "flag_baja_conf_votos": 0,
        "flag_caracter_anomalo": 0,
        "flag_descuadre_aritmetico": 0,
        "flag_omr_recuento_ambiguo": 0,
        "flag_sin_firmas_jurados": 0,
    }

    if df_e14.empty:
        print("[AVISO] No se encontraron archivos PDF para procesar.")
        return {"metrics": metrics, "flag_counts": flag_counts}

    # 3. Precarga de catálogo DIVIPOLE, estado de cola en BD y modelos neuronales
    supabase = create_client(supabase_url, supabase_key)
    stations_map = preload_stations_map(supabase)
    _, processed_files = get_pending_stations_map(supabase, stations_map)

    print("[FASE 2] Inicializando modelos TrOCR y EasyOCR en memoria...")
    device, processor, model, vocab_token_ids = load_trocr_resources()
    reader = load_easyocr_reader()

    # 4. Bucle principal por formulario PDF
    print("\n[=== INICIO DE PROCESAMIENTO POR LOTES ===]")
    for idx, (_, row) in enumerate(df_e14.iterrows(), start=1):
        pdf_path = row["path"]
        source_file = Path(pdf_path).stem

        # Omisión rápida en memoria O(1) para actas ya completas
        if source_file in processed_files:
            metrics["skipped"] += 1
            print(f"[{idx}/{total_files}] [SALTADO] '{source_file}' ya completo en Supabase.")
            continue

        print(f"[{idx}/{total_files}] [PROCESANDO] {source_file}...")

        # Paso A: Rasterización y alineación geométrica de ambas páginas
        alignment_error_detail = "No se detectaron marcas fiduciarias o PDF < 2 páginas"
        try:
            rois_list = process_e14_form(pdf_path, debug=False)
        except Exception as err:
            rois_list = None
            alignment_error_detail = str(err)

        if rois_list is None:
            try:
                res = insert_ocr_result(
                    supabase, None, source_file, station_id=None, alignment_failed=True
                )
                if res.get("status") == "skipped":
                    metrics["skipped"] += 1
                else:
                    metrics["failed_total"] += 1
                    flag_counts["err_alineacion"] += 1
                    log_processing_error(
                        errors_csv_path, source_file, "err_alineacion", alignment_error_detail
                    )
            except Exception as err:
                metrics["failed_unhandled"] += 1
                log_processing_error(errors_csv_path, source_file, "Fallo_RPC_Supabase", str(err))
                print(f"  [!] Error crítico al registrar fallo de alineación: {err}\n{traceback.format_exc()}")
            continue

        # Paso B: Extracción OCR / HTR / OMR sobre los recortes
        ocr_data, divipole_p1 = extract_e14_ocr_data(
            rois_list, reader, device, vocab_token_ids, processor, model
        )

        if ocr_data is None:
            try:
                res = insert_ocr_result(
                    supabase, None, source_file, station_id=None, ocr_failed=True
                )
                if res.get("status") == "skipped":
                    metrics["skipped"] += 1
                else:
                    metrics["failed_total"] += 1
                    flag_counts["err_inferencia_ocr"] += 1
                    log_processing_error(
                        errors_csv_path, source_file, "err_inferencia_ocr", "Excepción en OCR/HTR/OMR"
                    )
            except Exception as err:
                metrics["failed_unhandled"] += 1
                log_processing_error(errors_csv_path, source_file, "Fallo_RPC_Supabase", str(err))
                print(f"  [!] Error crítico al registrar fallo OCR: {err}\n{traceback.format_exc()}")
            continue

        # Paso C: Resolución de llave foránea y persistencia atómica en Supabase
        station_id = resolve_station_id(stations_map, divipole_p1)

        try:
            res = insert_ocr_result(supabase, ocr_data, source_file, station_id=station_id)
            status = res.get("status")

            if status == "skipped":
                metrics["skipped"] += 1
                continue

            active_flags = [k for k, v in res.get("flags", {}).items() if v]
            flag_details = res.get("flag_details", {})

            for flag_name in active_flags:
                flag_counts[flag_name] += 1
                log_processing_error(
                    errors_csv_path,
                    source_file,
                    flag_name,
                    flag_details.get(flag_name, f"id_divipole={res.get('id_divipole')}"),
                )

            if active_flags:
                metrics["inserted_with_flags"] += 1
            else:
                metrics["inserted_clean"] += 1

        except Exception as err:
            metrics["failed_unhandled"] += 1
            tb_str = traceback.format_exc()
            log_processing_error(errors_csv_path, source_file, "Fallo_RPC_Supabase", str(err))
            print(f"  [!] Error crítico de persistencia: {err}\n{tb_str}")

    # 5. Impresión del reporte consolidado de balance y frecuencias
    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print("RESUMEN DE EJECUCIÓN - FASE 2: VISIÓN & OCR PIPELINE")
    print("=" * 65)
    print(f"Tiempo total                 : {elapsed:.2f} s ({elapsed / 60:.2f} min)")
    print(f"Total formularios indexados  : {total_files}")
    print(f"  - Limpios (0 banderas)     : {metrics['inserted_clean']}")
    print(f"  - Con banderas de revisión : {metrics['inserted_with_flags']}")
    print(f"  - Fallos totales (solo DB) : {metrics['failed_total']}")
    print(f"  - Omitidos (ya en BD)      : {metrics['skipped']}")
    print(f"  - Errores críticos RPC     : {metrics['failed_unhandled']}")
    print("-" * 65)
    print("DESGLOSE POR BANDERA DE REVISIÓN (12 CASOS DE USO):")
    for flag_name, count in flag_counts.items():
        print(f"  - {flag_name:<30}: {count}")
    print("=" * 65 + "\n")

    return {"metrics": metrics, "flag_counts": flag_counts}