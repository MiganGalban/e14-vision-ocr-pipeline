from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import re

from src.processing.preprocess import pdf_to_images


def extract_common_rois(aligned_img: np.ndarray) -> dict[str, np.ndarray]:
    """
    Extrae los recortes (ROIs) del encabezado común presentes en ambas páginas del E-14.

    Args:
        aligned_img (np.ndarray): Imagen de página alineada y binarizada.

    Returns:
        dict[str, np.ndarray]: Diccionario con el nombre de la región y su matriz recortada.
    """
    height, width = aligned_img.shape[:2]
    
    #Se definen los ROI en porcentajes, para que sean proporciones.
    coords_map = {
        "Metadatos_Encabezado": (int(height * 0.026), int(height * 0.12), int(width * 0.242), width),
        "QR": (int(height * 0.04), int(height * 0.12), 0, int(width * 0.242)),
        "Barras": (int(height * 0.024), int(height * 0.038), int(width * 0.35), int(width * 0.65)),
        "Divipole": (int(height * 0.123), int(height * 0.186), 0, width),
        "Codigo": (int(height * 0.195), int(height * 0.215), int(width * 0.31), int(width * 0.69)),
    }

    return {
        zone: aligned_img[ymin:ymax, xmin:xmax]
        for zone, (ymin, ymax, xmin, xmax) in coords_map.items()
    }
    

def extract_page1_rois(aligned_img: np.ndarray) -> dict[str, np.ndarray]:
    """
    Extrae las casillas individuales de dígitos manuscritos de la Página 1 del E-14.

    Args:
        aligned_img (np.ndarray): Imagen de la Página 1 alineada y binarizada.

    Returns:
        dict[str, np.ndarray]: Diccionario con cada casilla individual (ej. 'Votos_A_1') y su recorte.
    """
    height, width = aligned_img.shape[:2]
    col1_x1, col1_x2 = int(width * 0.725), int(width * 0.825) #Celda vertical 1 
    col2_x1, col2_x2 = int(width * 0.810), int(width * 0.910) #Celda vertical 2 
    col3_x1, col3_x2 = int(width * 0.890), int(width * 0.990) #Celda vertical 3

    coords_map = {
        "Total_E11_1": (int(height * 0.24), int(height * 0.27), col1_x1, col1_x2),
        "Total_E11_2": (int(height * 0.24), int(height * 0.27), col2_x1, col2_x2),
        "Total_E11_3": (int(height * 0.24), int(height * 0.27), col3_x1, col3_x2),
        "Total_Urna_1": (int(height * 0.27), int(height * 0.30), col1_x1, col1_x2),
        "Total_Urna_2": (int(height * 0.27), int(height * 0.30), col2_x1, col2_x2),
        "Total_Urna_3": (int(height * 0.27), int(height * 0.30), col3_x1, col3_x2),
        "Total_Incinerados_1": (int(height * 0.30), int(height * 0.33), col1_x1, col1_x2),
        "Total_Incinerados_2": (int(height * 0.30), int(height * 0.33), col2_x1, col2_x2),
        "Total_Incinerados_3": (int(height * 0.30), int(height * 0.33), col3_x1, col3_x2),
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
    """
    Extrae los bloques completos de 3 dígitos por fila de la Página 1 para inspección.

    Args:
        aligned_img (np.ndarray): Imagen de la Página 1 alineada y binarizada.

    Returns:
        dict[str, np.ndarray]: Diccionario con el nombre del campo consolidado y su recorte.
    """
    height, width = aligned_img.shape[:2]
    col1_x1 = int(width * 0.72) #Celda vertical 1 - Primer limite
    col3_x1 = int(width * 0.89) #Celda vertical 3 - Primer limite
    col3_x2 = int(width * 0.99) 

    coords_map = {
        "Nivelacion_Mesa": (int(height * 0.24), int(height * 0.34), 0, col3_x2),
        "Total_E11": (int(height * 0.24), int(height * 0.27), col1_x1, col3_x2),
        "Total_Urna": (int(height * 0.27), int(height * 0.30), col1_x1, col3_x2),
        "Total_Incinerados": (int(height * 0.30), int(height * 0.33), col1_x1, col3_x2),
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
    """
    Extrae las regiones de verificación OMR y firmas de jurados de la Página 2 del E-14.

    Args:
        aligned_img (np.ndarray): Imagen de la Página 2 alineada y binarizada.

    Returns:
        dict[str, np.ndarray]: Diccionario con el nombre de la zona OMR/firma y su recorte.
    """

    height, width = aligned_img.shape[:2]
    
    coords_map = {
        "Observ_jud": (int(height * 0.22), int(height * 0.59), 0, width),
        "Re_count_si": (int(height * 0.598), int(height * 0.614), int(0.422 * width), int(0.480 * width)),
        "Re_count_no": (int(height * 0.598), int(height * 0.614), int(0.510 * width), int(0.568 * width)),
        "Count_req_by": (int(height * 0.621), int(height * 0.689), 0, width),
        "On_behalf_of": (int(height * 0.689), int(height * 0.756), 0, width),
        "Firma_J1": (int(height * 0.780), int(height * 0.823), 0, int(0.48 * width)),
        "Firma_J3": (int(height * 0.851), int(height * 0.894), 0, int(0.48 * width)),
        "Firma_J5": (int(height * 0.923), int(height * 0.966), 0, int(0.48 * width)),
        "Firma_J2": (int(height * 0.780), int(height * 0.823), int(0.50 * width), int(0.98 * width)),
        "Firma_J4": (int(height * 0.851), int(height * 0.894), int(0.50 * width), int(0.98 * width)),
        "Firma_J6": (int(height * 0.923), int(height * 0.966), int(0.50 * width), int(0.98 * width)),
        "Firma_cc_J1": (int(height * 0.823), int(height * 0.838), int(0.065 * width), int(0.48 * width)),
        "Firma_cc_J3": (int(height * 0.894), int(height * 0.909), int(0.065 * width), int(0.48 * width)),
        "Firma_cc_J5": (int(height * 0.966), int(height * 0.981), int(0.065 * width), int(0.48 * width)),
        "Firma_cc_J2": (int(height * 0.823), int(height * 0.838), int(0.565 * width), int(0.98 * width)),
        "Firma_cc_J4": (int(height * 0.894), int(height * 0.909), int(0.565 * width), int(0.98 * width)),
        "Firma_cc_J6": (int(height * 0.966), int(height * 0.981), int(0.565 * width), int(0.98 * width)),
    }

    return {
        zone: aligned_img[ymin:ymax, xmin:xmax]
        for zone, (ymin, ymax, xmin, xmax) in coords_map.items()
    }

# ---------------------------------------------------------------------------# ---------------------------------------------------------------------------

def parse_divipole_text(ocr_items: list[dict[str, str | float]]) -> dict[str, str | None]:
    """
    Extrae variables estructuradas de ubicación geográfica a partir de la salida de EasyOCR.

    Contempla sustitución de 'O'/'o' por '0' en campos estrictamente numéricos y soporta
    códigos alfanuméricos en puestos de votación (ej. 'A1', 'B3').

    Args:
        ocr_items (list[dict[str, str | float]]): Lista de diccionarios generada por
            `read_text_region` con llaves 'text' (o 'texto') y 'score'.

    Returns:
        dict[str, str | None]: Diccionario con las llaves DP_Dept, DP_Munp, DP_Zona,
        DP_Psto, DP_Mesa y DP_Lugr.
    """
    
    parsed_vars: dict[str, str | None] = {
        "DP_Dept": None,
        "DP_Munp": None,
        "DP_Zona": None,
        "DP_Psto": None,
        "DP_Mesa": None,
        "DP_Lugr": None, # Captura desde "LUGAR:" hasta el final
    }

    patterns = {
        "DP_Dept": r"DEPARTAMENTO:\s*([\doO]+)",
        "DP_Munp": r"MUNICIPIO:\s*([\doO]+)",
        "DP_Zona": r"ZONA:\s*([\doO]+)",
        "DP_Psto": r"PUESTO:\s*([a-zA-Z\d]+)",
        "DP_Mesa": r"MESA:\s*([\doO]+)",
        "DP_Lugr": r"LUGAR:\s*(.+?)(?=$)",
    }

    numeric_keys = {"DP_Dept", "DP_Munp", "DP_Zona", "DP_Mesa"}

    for item in ocr_items:
        raw_text = str(item.get("text") or item.get("texto") or "")

        for key, pattern in patterns.items():
            # Si la variable ya fue encontrada, no la sobrescribe
            if parsed_vars[key] is not None:
                continue
                
            match = re.search(pattern, raw_text, re.IGNORECASE)
            
            if match:
                captured_val = match.group(1).strip()
                
                # Coerción numérica
                if key in numeric_keys:
                    parsed_vars[key] = captured_val.upper().replace("O", "0")
                elif key == "DP_Psto":
                    parsed_vars[key] = captured_val.upper()
                else:
                    parsed_vars[key] = captured_val
                break # Rompe el ciclo de patrones si ya encontró coincidencia para este item
                
    return parsed_vars

def clean_float(val: float | int | np.floating | None, precision: int = 2) -> float:
    """Convierte un valor numérico o numpy.float a float nativo redondeado.

    Args:
        val (float | int | np.floating | None): Valor numérico a normalizar.
        precision (int, opcional): Número de decimales a conservar. Por defecto es 2.

    Returns:
        float: Valor flotante redondeado, o 0.0 si la entrada es None.
    """
    if val is None:
        return 0.0
    return round(float(val), precision)


def consolidate_digits(d1: str, d2: str, d3: str) -> int:
    """Consolida las predicciones de 3 casillas individuales en un número entero.

    Ignora etiquetas como 'EN_BLANCO' o símbolos especiales, conservando únicamente
    los caracteres numéricos.

    Args:
        d1 (str): Carácter predicho para las centenas (casilla 1).
        d2 (str): Carácter predicho para las decenas (casilla 2).
        d3 (str): Carácter predicho para las unidades (casilla 3).

    Returns:
        int: Valor entero consolidado, o 0 si ninguna casilla contiene dígitos.
    """
    raw_text = f"{d1}{d2}{d3}".strip()
    digits = "".join(filter(str.isdigit, raw_text))
    return int(digits) if digits else 0


def normalize_code(value: str | int | None, length: int) -> str:
    """Estandariza códigos DIVIPOLE aplicando mayúsculas y relleno de ceros a la izquierda.

    Args:
        value (str | int | None): Código original a normalizar.
        length (int): Longitud objetivo del código cuando es numérico (ej. 2 o 3).

    Returns:
        str: Código limpio y normalizado, o cadena vacía si la entrada es None.
    """
    if value is None:
        return ""
    val_str = str(value).strip().upper()
    if val_str.isdigit():
        return val_str.zfill(length)
    return val_str


def scan_e14_pdfs(root_dir: str | Path) -> pd.DataFrame:
    """Escanea recursivamente un directorio en busca de formularios PDF E-14 válidos.

    Extrae las llaves estructurales (dd, mm, zz, pp, mesa) a partir del nombre de archivo.

    Args:
        root_dir (str | Path): Ruta del directorio base a escanear.

    Raises:
        FileNotFoundError: Si la ruta especificada no existe en el sistema de archivos.

    Returns:
        pd.DataFrame: DataFrame con columnas ['dd', 'mm', 'zz', 'pp', 'mesa', 'path'].
    """
    base_path = Path(root_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"La ruta especificada no existe: {root_dir}")

    e14_pattern = re.compile(
        r"^(?P<dd>\d{2})_(?P<mm>\d{3})_(?P<zz>\d{2})_(?P<pp>[a-zA-Z0-9]{2})_(?P<mesa>\d{3,4})\.pdf$",
        re.IGNORECASE,
    )

    records = []
    for file_path in sorted(base_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() == ".pdf":
            match = e14_pattern.match(file_path.name)
            if match:
                record = match.groupdict()
                record["path"] = str(file_path.resolve())
                records.append(record)

    df = pd.DataFrame(records)
    if not df.empty:
        df = df[["dd", "mm", "zz", "pp", "mesa", "path"]]

    return df

def process_e14_form(pdf_path: str | Path, debug: bool = False) -> list[dict[str, np.ndarray]] | None:
    """Convierte un PDF E-14 en imágenes alineadas y extrae los diccionarios de ROIs de ambas páginas.

    Args:
        pdf_path (str | Path): Ruta del archivo PDF a procesar.
        debug (bool, opcional): Activa visualización de depuración geométrica. Por defecto es False.

    Returns:
        list[dict[str, np.ndarray]] | None: Lista con 4 diccionarios de recortes
        [encabezado_p1, encabezado_p2, casillas_p1, omr_p2], o None si falla la alineación.
    """
    aligned_pages = pdf_to_images(pdf_path, debug=debug)
    if len(aligned_pages) < 2 or aligned_pages[0] is None or aligned_pages[1] is None:
        return None

    img_page_1, img_page_2 = aligned_pages[0], aligned_pages[1]
    return [
        extract_common_rois(img_page_1),
        extract_common_rois(img_page_2),
        extract_page1_rois(img_page_1),
        extract_page2_rois(img_page_2),
    ]
