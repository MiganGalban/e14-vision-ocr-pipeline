from pathlib import Path
import re

import numpy as np
import pandas as pd

from src.processing.preprocess import pdf_to_images


def extract_common_rois(aligned_img: np.ndarray) -> dict[str, np.ndarray]:
    """Extrae los recortes (ROIs) del encabezado común presentes en ambas páginas del E-14.

    Args:
        aligned_img (np.ndarray): Imagen de página alineada y binarizada.

    Returns:
        dict[str, np.ndarray]: Diccionario con el nombre de cada región y su matriz recortada.
    """
    height, width = aligned_img.shape[:2]

    # Coordenadas proporcionales (ymin, ymax, xmin, xmax) sobre la página normalizada
    coords_map = {
        "Metadatos_Encabezado": (int(height * 0.026), int(height * 0.120), int(width * 0.242), width),
        "QR": (int(height * 0.040), int(height * 0.120), 0, int(width * 0.242)),
        "Barras": (int(height * 0.024), int(height * 0.038), int(width * 0.350), int(width * 0.650)),
        "Divipole": (int(height * 0.123), int(height * 0.186), 0, width),
        "Codigo": (int(height * 0.195), int(height * 0.215), int(width * 0.310), int(width * 0.690)),
    }

    # Segmentación matricial de cada región definida
    return {
        zone: aligned_img[ymin:ymax, xmin:xmax]
        for zone, (ymin, ymax, xmin, xmax) in coords_map.items()
    }


def extract_page1_rois(aligned_img: np.ndarray) -> dict[str, np.ndarray]:
    """Extrae las 27 casillas individuales de dígitos manuscritos de la Página 1 del E-14.

    Args:
        aligned_img (np.ndarray): Imagen de la Página 1 alineada y binarizada.

    Returns:
        dict[str, np.ndarray]: Diccionario con cada casilla (ej. 'Votos_A_1') y su recorte.
    """
    height, width = aligned_img.shape[:2]

    # Límites horizontales de las 3 columnas de dígitos (centenas, decenas y unidades)
    col1_x1, col1_x2 = int(width * 0.725), int(width * 0.825)
    col2_x1, col2_x2 = int(width * 0.810), int(width * 0.910)
    col3_x1, col3_x2 = int(width * 0.890), int(width * 0.990)

    # Mapa de coordenadas por fila de nivelación, candidatos y totales
    coords_map = {
        "Total_E11_1": (int(height * 0.240), int(height * 0.270), col1_x1, col1_x2),
        "Total_E11_2": (int(height * 0.240), int(height * 0.270), col2_x1, col2_x2),
        "Total_E11_3": (int(height * 0.240), int(height * 0.270), col3_x1, col3_x2),
        "Total_Urna_1": (int(height * 0.270), int(height * 0.300), col1_x1, col1_x2),
        "Total_Urna_2": (int(height * 0.270), int(height * 0.300), col2_x1, col2_x2),
        "Total_Urna_3": (int(height * 0.270), int(height * 0.300), col3_x1, col3_x2),
        "Total_Incinerados_1": (int(height * 0.300), int(height * 0.330), col1_x1, col1_x2),
        "Total_Incinerados_2": (int(height * 0.300), int(height * 0.330), col2_x1, col2_x2),
        "Total_Incinerados_3": (int(height * 0.300), int(height * 0.330), col3_x1, col3_x2),
        "Votos_A_1": (int(height * 0.445), int(height * 0.475), col1_x1, col1_x2),
        "Votos_A_2": (int(height * 0.445), int(height * 0.475), col2_x1, col2_x2),
        "Votos_A_3": (int(height * 0.445), int(height * 0.475), col3_x1, col3_x2),
        "Votos_B_1": (int(height * 0.605), int(height * 0.635), col1_x1, col1_x2),
        "Votos_B_2": (int(height * 0.605), int(height * 0.635), col2_x1, col2_x2),
        "Votos_B_3": (int(height * 0.605), int(height * 0.635), col3_x1, col3_x2),
        "V_Blanco_1": (int(height * 0.723), int(height * 0.753), col1_x1, col1_x2),
        "V_Blanco_2": (int(height * 0.723), int(height * 0.753), col2_x1, col2_x2),
        "V_Blanco_3": (int(height * 0.723), int(height * 0.753), col3_x1, col3_x2),
        "V_Nulos_1": (int(height * 0.753), int(height * 0.783), col1_x1, col1_x2),
        "V_Nulos_2": (int(height * 0.753), int(height * 0.783), col2_x1, col2_x2),
        "V_Nulos_3": (int(height * 0.753), int(height * 0.783), col3_x1, col3_x2),
        "V_Sin_Marcar_1": (int(height * 0.785), int(height * 0.815), col1_x1, col1_x2),
        "V_Sin_Marcar_2": (int(height * 0.785), int(height * 0.815), col2_x1, col2_x2),
        "V_Sin_Marcar_3": (int(height * 0.785), int(height * 0.815), col3_x1, col3_x2),
        "Suma_Total_1": (int(height * 0.817), int(height * 0.847), col1_x1, col1_x2),
        "Suma_Total_2": (int(height * 0.817), int(height * 0.847), col2_x1, col2_x2),
        "Suma_Total_3": (int(height * 0.817), int(height * 0.847), col3_x1, col3_x2),
    }

    return {
        zone: aligned_img[ymin:ymax, xmin:xmax]
        for zone, (ymin, ymax, xmin, xmax) in coords_map.items()
    }


def extract_page1_row_rois(aligned_img: np.ndarray) -> dict[str, np.ndarray]:
    """Extrae los bloques completos de 3 dígitos por fila de la Página 1 para inspección visual.

    Args:
        aligned_img (np.ndarray): Imagen de la Página 1 alineada y binarizada.

    Returns:
        dict[str, np.ndarray]: Diccionario con el nombre del campo consolidado y su recorte.
    """
    height, width = aligned_img.shape[:2]

    # Rango horizontal que abarca las 3 casillas de votación conjuntas
    col1_x1 = int(width * 0.720)
    col3_x2 = int(width * 0.990)

    coords_map = {
        "Nivelacion_Mesa": (int(height * 0.240), int(height * 0.340), 0, col3_x2),
        "Total_E11": (int(height * 0.240), int(height * 0.270), col1_x1, col3_x2),
        "Total_Urna": (int(height * 0.270), int(height * 0.300), col1_x1, col3_x2),
        "Total_Incinerados": (int(height * 0.300), int(height * 0.330), col1_x1, col3_x2),
        "Votos_A": (int(height * 0.445), int(height * 0.475), col1_x1, col3_x2),
        "Votos_B": (int(height * 0.605), int(height * 0.635), col1_x1, col3_x2),
        "V_Blanco": (int(height * 0.723), int(height * 0.753), col1_x1, col3_x2),
        "V_Nulos": (int(height * 0.753), int(height * 0.783), col1_x1, col3_x2),
        "V_Sin_Marcar": (int(height * 0.785), int(height * 0.815), col1_x1, col3_x2),
        "Suma_Total": (int(height * 0.817), int(height * 0.847), col1_x1, col3_x2),
    }

    return {
        zone: aligned_img[ymin:ymax, xmin:xmax]
        for zone, (ymin, ymax, xmin, xmax) in coords_map.items()
    }


def extract_page2_rois(aligned_img: np.ndarray) -> dict[str, np.ndarray]:
    """Extrae las regiones de verificación OMR, recuento y firmas de jurados de la Página 2.

    Args:
        aligned_img (np.ndarray): Imagen de la Página 2 alineada y binarizada.

    Returns:
        dict[str, np.ndarray]: Diccionario con el nombre de cada zona OMR/firma y su recorte.
    """
    height, width = aligned_img.shape[:2]

    # Coordenadas proporcionales de constancias, checkboxes de recuento, firmas y cédulas
    coords_map = {
        "Observ_jud": (int(height * 0.220), int(height * 0.590), 0, width),
        "Re_count_si": (int(height * 0.598), int(height * 0.614), int(width * 0.422), int(width * 0.480)),
        "Re_count_no": (int(height * 0.598), int(height * 0.614), int(width * 0.510), int(width * 0.568)),
        "Count_req_by": (int(height * 0.621), int(height * 0.689), 0, width),
        "On_behalf_of": (int(height * 0.689), int(height * 0.756), 0, width),
        "Firma_J1": (int(height * 0.780), int(height * 0.823), 0, int(width * 0.480)),
        "Firma_J3": (int(height * 0.851), int(height * 0.894), 0, int(width * 0.480)),
        "Firma_J5": (int(height * 0.923), int(height * 0.966), 0, int(width * 0.480)),
        "Firma_J2": (int(height * 0.780), int(height * 0.823), int(width * 0.500), int(width * 0.980)),
        "Firma_J4": (int(height * 0.851), int(height * 0.894), int(width * 0.500), int(width * 0.980)),
        "Firma_J6": (int(height * 0.923), int(height * 0.966), int(width * 0.500), int(width * 0.980)),
        "Firma_cc_J1": (int(height * 0.823), int(height * 0.838), int(width * 0.065), int(width * 0.480)),
        "Firma_cc_J3": (int(height * 0.894), int(height * 0.909), int(width * 0.065), int(width * 0.480)),
        "Firma_cc_J5": (int(height * 0.966), int(height * 0.981), int(width * 0.065), int(width * 0.480)),
        "Firma_cc_J2": (int(height * 0.823), int(height * 0.838), int(width * 0.565), int(width * 0.980)),
        "Firma_cc_J4": (int(height * 0.894), int(height * 0.909), int(width * 0.565), int(width * 0.980)),
        "Firma_cc_J6": (int(height * 0.966), int(height * 0.981), int(width * 0.565), int(width * 0.980)),
    }

    return {
        zone: aligned_img[ymin:ymax, xmin:xmax]
        for zone, (ymin, ymax, xmin, xmax) in coords_map.items()
    }


def parse_divipole_text(ocr_items: list[dict[str, str | float]]) -> dict[str, str | None]:
    """Extrae los campos geográficos estructurados DIVIPOLE a partir de la salida de EasyOCR.

    Sustituye 'O'/'o' por '0' en los campos estrictamente numéricos y conserva valores
    alfanuméricos en puestos especiales (ej. 'A1', 'B3').

    Args:
        ocr_items (list[dict[str, str | float]]): Lista de diccionarios generada por
            `read_text_region` con llaves 'text' y 'score'.

    Returns:
        dict[str, str | None]: Diccionario con DP_Dept, DP_Munp, DP_Zona, DP_Psto, DP_Mesa y DP_Lugr.
    """
    parsed_vars: dict[str, str | None] = {
        "DP_Dept": None,
        "DP_Munp": None,
        "DP_Zona": None,
        "DP_Psto": None,
        "DP_Mesa": None,
        "DP_Lugr": None,
    }

    # Patrones de captura por campo (DP_Lugr captura hasta el final de la línea)
    patterns = {
        "DP_Dept": r"DEPARTAMENTO:\s*([\doO]+)",
        "DP_Munp": r"MUNICIPIO:\s*([\doO]+)",
        "DP_Zona": r"ZONA:\s*([\doO]+)",
        "DP_Psto": r"PUESTO:\s*([a-zA-Z\d]+)",
        "DP_Mesa": r"MESA:\s*([\doO]+)",
        "DP_Lugr": r"LUGAR:\s*(.+?)(?=$)",
    }
    numeric_keys = {"DP_Dept", "DP_Munp", "DP_Zona", "DP_Mesa"}

    # Se evalúan todos los patrones en cada línea para soportar campos agrupados (ej. ZONA + PUESTO + MESA)
    for item in ocr_items:
        raw_text = str(item.get("text") or item.get("texto") or "")

        for key, pattern in patterns.items():
            if parsed_vars[key] is not None:
                continue

            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                captured_val = match.group(1).strip()
                if key in numeric_keys:
                    parsed_vars[key] = captured_val.upper().replace("O", "0")
                elif key == "DP_Psto":
                    parsed_vars[key] = captured_val.upper()
                else:
                    parsed_vars[key] = captured_val

    return parsed_vars


def clean_float(val: float | int | np.floating | None, precision: int = 2) -> float:
    """Convierte un escalar numérico o numpy.float a float nativo de Python redondeado.

    Args:
        val (float | int | np.floating | None): Valor numérico a normalizar.
        precision (int, opcional): Cantidad de decimales a conservar. Por defecto es 2.

    Returns:
        float: Valor flotante redondeado, o 0.0 si la entrada es None.
    """
    if val is None:
        return 0.0
    return round(float(val), precision)


def consolidate_digits(d1: str, d2: str, d3: str) -> int:
    """Consolida las predicciones de 3 casillas individuales en un entero no negativo.

    Descarta etiquetas como 'EN_BLANCO' y símbolos especiales, conservando solo dígitos.

    Args:
        d1 (str): Carácter predicho en la casilla de centenas.
        d2 (str): Carácter predicho en la casilla de decenas.
        d3 (str): Carácter predicho en la casilla de unidades.

    Returns:
        int: Número entero consolidado, o 0 si ninguna casilla contiene dígitos.
    """
    raw_text = f"{d1}{d2}{d3}".strip()
    digits = "".join(filter(str.isdigit, raw_text))
    return int(digits) if digits else 0


def normalize_code(value: str | int | None, length: int) -> str:
    """Estandariza códigos DIVIPOLE en mayúsculas y aplica relleno de ceros a la izquierda.

    Args:
        value (str | int | None): Código original a normalizar.
        length (int): Longitud objetivo cuando el código es puramente numérico (ej. 2 o 3).

    Returns:
        str: Código limpio normalizado, o cadena vacía si la entrada es None.
    """
    if value is None:
        return ""
    val_str = str(value).strip().upper()
    if val_str.isdigit():
        return val_str.zfill(length)
    return val_str


def scan_e14_pdfs(root_dir: str | Path) -> pd.DataFrame:
    """Escanea recursivamente un directorio en busca de formularios PDF E-14 válidos.

    Extrae las llaves estructurales (dd, mm, zz, pp, mesa) a partir de la nomenclatura del archivo.

    Args:
        root_dir (str | Path): Ruta base del directorio a escanear.

    Raises:
        FileNotFoundError: Si el directorio especificado no existe.

    Returns:
        pd.DataFrame: DataFrame con columnas ['dd', 'mm', 'zz', 'pp', 'mesa', 'path'].
    """
    base_path = Path(root_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"La ruta especificada no existe: {root_dir}")

    # Patrón estricto de archivo: dd(2)_mm(3)_zz(2)_pp(2 alfanum)_mesa(3-4).pdf
    e14_pattern = re.compile(
        r"^(?P<dd>\d{2})_(?P<mm>\d{3})_(?P<zz>\d{2})_(?P<pp>[a-zA-Z0-9]{2})_(?P<mesa>\d{3,4})\.pdf$",
        re.IGNORECASE,
    )

    # Recorrido ordenado y extracción de llaves por archivo válido
    records = []
    for file_path in sorted(base_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() == ".pdf":
            match = e14_pattern.match(file_path.name)
            if match:
                record = match.groupdict()
                record["path"] = str(file_path.resolve())
                records.append(record)

    columns = ["dd", "mm", "zz", "pp", "mesa", "path"]
    if not records:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(records)[columns]


def process_e14_form(pdf_path: str | Path, debug: bool = False) -> list[dict[str, np.ndarray]] | None:
    """Rasteriza y alinea un PDF E-14 para extraer los 4 diccionarios ROI de ambas páginas.

    Args:
        pdf_path (str | Path): Ruta local del archivo PDF a segmentar.
        debug (bool, opcional): Activa visualizaciones de diagnóstico durante la alineación.
            Por defecto es False.

    Returns:
        list[dict[str, np.ndarray]] | None: Lista de 4 diccionarios de recortes
        [encabezado_p1, encabezado_p2, casillas_p1, omr_p2], o None si falla la alineación.
    """
    # Conversión y validación geométrica de las 2 páginas obligatorias del acta
    aligned_pages = pdf_to_images(pdf_path, debug=debug)
    if len(aligned_pages) < 2 or aligned_pages[0] is None or aligned_pages[1] is None:
        return None

    img_page_1, img_page_2 = aligned_pages[0], aligned_pages[1]

    # Segmentación de encabezados comunes, casillas manuscritas (Pág. 1) y zonas OMR (Pág. 2)
    return [
        extract_common_rois(img_page_1),
        extract_common_rois(img_page_2),
        extract_page1_rois(img_page_1),
        extract_page2_rois(img_page_2),
    ]