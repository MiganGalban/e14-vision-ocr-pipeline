import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import numpy as np
from postgrest.exceptions import APIError
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


def preload_stations_map(supabase: Client) -> dict[tuple[str, str, str, str], int]:
    """
    Descarga la tabla divipole_regis paginada y construye un mapa hash O(1) en memoria.

    Args:
        supabase (Client): Cliente autenticado de Supabase.

    Returns:
        dict[tuple[str, str, str, str], int]: Mapa {(dd, mm, zz, pp): puesto_id}.
    """
    records = []
    batch_size = 1000
    start = 0

    while True:
        end = start + batch_size - 1
        resp = ( supabase.table("divipole_regis")
                .select("id, dd, mm, zz, pp").neq("dd", "88").range(start, end).execute() )
        
        rows = resp.data
        if not rows:
            break

        records.extend(rows)
        if len(rows) < batch_size:
            break
        start += batch_size

    stations_map: dict[tuple[str, str, str, str], int] = {}
    for row in records:
        key = ( normalize_code(row["dd"], 2), normalize_code(row["mm"], 3),
                normalize_code(row["zz"], 2), normalize_code(row["pp"], 2),  )
        
        stations_map[key] = int(row["id"])

    print(f"[CACHE] Mapa de puestos cargado en memoria: {len(stations_map):,} puestos indexados.")
    return stations_map


def get_pending_stations_map( supabase: Client, stations_map: dict[tuple[str, str, str, str], int],
                                ) -> tuple[dict[tuple[str, str, str, str], int], set[str]]:
    """
    Consulta los registros ya cargados en 'divipole' para identificar puestos y archivos pendientes.

    Args:
        supabase (Client): Cliente autenticado de Supabase.
        stations_map (dict[tuple[str, str, str, str], int]): Mapa maestro de puestos.

    Returns:
        tuple: (mapa de puestos pendientes, conjunto de nombres de archivo_origen ya procesados).
    """
    divipole_records = []
    batch_size = 1000
    start = 0

    while True:
        end = start + batch_size - 1
        resp = ( supabase.table("divipole")
                .select("puesto_id, archivo_origen").range(start, end).execute() )
        
        rows = resp.data
        if not rows:
            break

        divipole_records.extend(rows)
        if len(rows) < batch_size:
            break
        start += batch_size

    processed_station_ids = {
        r["puesto_id"] for r in divipole_records if r.get("puesto_id") is not None
    }
    processed_files = {
        str(r["archivo_origen"]) for r in divipole_records if r.get("archivo_origen") is not None
    }

    pending_stations_map = {
        key: pid for key, pid in stations_map.items() if pid not in processed_station_ids
    }

    print("[ESTADO DE COLA]")
    print(f" - Formularios procesados en BD: {len(divipole_records):,}")
    print(f" - Puestos con registros existentes: {len(processed_station_ids):,}")
    print(f" - Puestos totalmente pendientes: {len(pending_stations_map):,}")

    return pending_stations_map, processed_files


def resolve_station_id( stations_map: dict[tuple[str, str, str, str], int],
                        ocr_divipole: dict[str, str | None], ) -> int | None:
    """
    Resuelve el puesto_id en tiempo O(1) a partir de los códigos leídos por OCR.

    Args:
        stations_map (dict[tuple[str, str, str, str], int]): Mapa maestro {(dd, mm, zz, pp): id}.
        ocr_divipole (dict[str, str | None]): Diccionario DIVIPOLE extraído del encabezado.

    Returns:
        int | None: ID primario del puesto en `divipole_regis`, o None si no existe coincidencia.
    """
    dd = normalize_code(ocr_divipole.get("DP_Dept"), 2)
    mm = normalize_code(ocr_divipole.get("DP_Munp"), 3)
    zz = normalize_code(ocr_divipole.get("DP_Zona"), 2)
    pp = normalize_code(ocr_divipole.get("DP_Psto"), 2)

    return stations_map.get((dd, mm, zz, pp))


def insert_ocr_result( supabase: Client, ocr_data: dict[str, Any] | None, source_file: str,
                        station_id: int | None = None, evidence_crops: dict[str, Any] | None = None,
                        ) -> dict[str, Any]:
    """
    Inserta los datos extraídos del formulario E-14 en el modelo relacional de Supabase.

    Si `ocr_data` es None (fallo geométrico o de lectura), registra el archivo en la tabla
    `divipole` con valores nulos y `req_revision = True`.

    Args:
        supabase (Client): Cliente autenticado de Supabase.
        ocr_data (dict[str, Any] | None): Diccionario consolidado retornado por `extract_e14_ocr_data`.
        source_file (str): Identificador del archivo sin extensión (ej. '01_001_02_02_029').
        station_id (int | None, opcional): Clave foránea hacia `divipole_regis`. Por defecto es None.
        evidence_crops (dict[str, Any] | None, opcional): Recortes serializados de respaldo.

    Returns:
        dict[str, Any]: Estado de la transacción y clave primaria `id_divipole` generada.
    """
    try:
        existing = ( supabase.table("divipole")
                    .select("id").eq("archivo_origen", source_file).execute() )
        
        if existing.data:
            print(f"[OMITIDO] '{source_file}' ya existe en la base de datos.")
            return {"status": "skipped", "id_divipole": existing.data[0]["id"]}

        if ocr_data is None:
            failed_payload = {
                "archivo_origen": source_file, 
                "puesto_id": station_id,
                "dp_mesa": None,    "dp_lugr": None,
                "cod_x": None,      "s_cod_x": None,
                "barras_1": None,   "s_barras_1": None, 
                "barras_2": None,   "s_barras_2": None,
                "req_revision": True,
            }
            
            res = supabase.table("divipole").insert(failed_payload).execute()
            id_divipole = res.data[0]["id"]
            print(f"[MANUAL] '{source_file}' registrado con fallo de lectura para revisión.")
            return {"status": "marked_for_manual_review", "id_divipole": id_divipole}

        evidence_crops = evidence_crops or {}
        
        dp_mesa_str = (
            str(ocr_data.get("DP_Mesa", "")).zfill(3) if ocr_data.get("DP_Mesa") else None
        )

        payload_divipole = {
            "puesto_id": station_id,
            "dp_mesa": dp_mesa_str,
            "archivo_origen": source_file,
            "dp_lugr": ocr_data.get("DP_Lugr"),
            "cod_x": str(ocr_data.get("Codigo")) if ocr_data.get("Codigo") else None,
            "s_cod_x": clean_float(ocr_data.get("s_codigo")),
            "barras_1": str(ocr_data.get("Barras_1")) if ocr_data.get("Barras_1") else None,
            "s_barras_1": clean_float(ocr_data.get("s_barras_1")),
            "barras_2": str(ocr_data.get("Barras_2")) if ocr_data.get("Barras_2") else None,
            "s_barras_2": clean_float(ocr_data.get("s_barras_2")),
            "req_revision": bool(ocr_data.get("manual", False)),
        }

        res_divipole = supabase.table("divipole").insert(payload_divipole).execute()
        id_divipole = res_divipole.data[0]["id"]

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

        consolidated_totals: dict[str, int] = {}
        votes_precision_payload: list[dict[str, Any]] = []
        vote_scores: list[float] = []

        for db_field, (k1, k2, k3) in votes_mapping.items():
            val_int = consolidate_digits(
                str(ocr_data.get(k1, "0")),
                str(ocr_data.get(k2, "0")),
                str(ocr_data.get(k3, "0")),
            )
            consolidated_totals[db_field] = val_int

            for idx, key in enumerate([k1, k2, k3], start=1):
                conf = clean_float(ocr_data.get(f"s_{key}", 0.0))
                vote_scores.append(conf)
                votes_precision_payload.append(
                    {
                        "campo_nombre": db_field,
                        "casilla_index": idx,
                        "prediccion_raw": str(ocr_data.get(key, "")),
                        "confianza": conf,
                        "data_recorte": evidence_crops.get(key),
                    }
                )

        avg_confidence = clean_float(np.mean(vote_scores)) if vote_scores else 0.0

        payload_votes = {
            "id_divipole": id_divipole,
            **consolidated_totals,
            "confianza_promedio_ocr": avg_confidence,
        }

        res_votes = supabase.table("votos").insert(payload_votes).execute()
        id_votes = res_votes.data[0]["id"]

        for item in votes_precision_payload:
            item["id_votos"] = id_votes
        supabase.table("votos_precision").insert(votes_precision_payload).execute()

        audit_mapping = {
            "observ_jud": "Observ_jud",     "re_count_si": "Re_count_si", 
            "re_count_no": "Re_count_no",   "count_req_by": "Count_req_by", 
            "on_behalf_of": "On_behalf_of",
            
            "firma_j1": "Firma_J1", "firma_j2": "Firma_J2", "firma_j3": "Firma_J3",
            "firma_j4": "Firma_J4", "firma_j5": "Firma_J5", "firma_j6": "Firma_J6",
            
            "firma_cc_j1": "Firma_cc_J1", "firma_cc_j2": "Firma_cc_J2",
            "firma_cc_j3": "Firma_cc_J3", "firma_cc_j4": "Firma_cc_J4",
            "firma_cc_j5": "Firma_cc_J5", "firma_cc_j6": "Firma_cc_J6",
        }

        payload_audit: dict[str, Any] = {"id_divipole": id_divipole}
        audit_precision_payload: list[dict[str, Any]] = []

        for db_field, dict_key in audit_mapping.items():
            omr_tuple = ocr_data.get(dict_key, (False, 0.0))
            payload_audit[db_field] = bool(omr_tuple[0])
            audit_precision_payload.append(
                {
                    "campo_nombre": db_field,
                    "valor_extraido": clean_float(omr_tuple[1], precision=2),
                }
            )

        res_audit = supabase.table("auditoria").insert(payload_audit).execute()
        id_audit = res_audit.data[0]["id"]

        for item in audit_precision_payload:
            item["id_auditoria"] = id_audit
        supabase.table("auditoria_precision").insert(audit_precision_payload).execute()

        return {"status": "success", "id_divipole": id_divipole}

    except APIError as err:
        if err.code == "23505":
            print(f"[DUPLICADO] Conflicto de clave única para '{source_file}'.")
            return {"status": "duplicate"}
        raise err


def execute_ocr_pipeline(pdf_root_dir: str | Path | None = None) -> None:
    """
    Orquesta el pipeline completo de Fase 2: escaneo de PDFs, inferencia OCR/HTR y carga a Supabase.

    Args:
        pdf_root_dir (str | Path | None, opcional): Directorio raíz donde se almacenan los PDFs
            descargados. Si es None, utiliza la carpeta `data/` en la raíz del repositorio.

    Raises:
        ValueError: Si las credenciales `SUPABASE_URL` o `SUPABASE_KEY` no están configuradas.
    """
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("Variables SUPABASE_URL o SUPABASE_KEY no definidas en el entorno.")

    if pdf_root_dir is None:
        pdf_root_dir = Path(__file__).resolve().parents[2] / "data"

    df_e14 = scan_e14_pdfs(pdf_root_dir)
    print(f"[FASE 2] Total formularios PDF encontrados: {len(df_e14)}")
    if df_e14.empty:
        print("[AVISO] No se encontraron archivos PDF con formato E-14 para procesar.")
        return

    supabase = create_client(supabase_url, supabase_key)
    stations_map = preload_stations_map(supabase)
    _, processed_files = get_pending_stations_map(supabase, stations_map)

    print("[FASE 2] Inicializando modelos TrOCR y EasyOCR en memoria...")
    device, processor, model, vocab_token_ids = load_trocr_resources()
    reader = load_easyocr_reader()

    for _, row in df_e14.iterrows():
        pdf_path = row["path"]
        source_file = Path(pdf_path).stem

        if source_file in processed_files:
            print(f"[SALTADO] '{source_file}' ya existe en Supabase.")
            continue

        print(f"[PROCESANDO] {source_file}...")
        rois_list = process_e14_form(pdf_path, debug=False)
        if rois_list is None:
            print(f"[ERROR GEOMÉTRICO] No se pudo alinear '{source_file}'.")
            insert_ocr_result(supabase, None, source_file, station_id=None)
            continue

        ocr_data, divipole_p1 = extract_e14_ocr_data(
            rois_list, reader, device, vocab_token_ids, processor, model
        )
        station_id = resolve_station_id(stations_map, divipole_p1) if ocr_data else None
        insert_ocr_result(supabase, ocr_data, source_file, station_id=station_id)