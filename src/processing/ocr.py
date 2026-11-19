from typing import Any

import cv2
import easyocr
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    AutoImageProcessor,
    RobertaTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from src.processing.tools import parse_divipole_text

DEFAULT_TROCR_MODEL = "microsoft/trocr-base-handwritten"
DEFAULT_VOCAB = "0123456789.*#-/\\"

# Umbrales OMR calibrados por tipo de zona: (porcentaje_minimo_tinta, margen_interior_pct)
OMR_ZONE_THRESHOLDS: dict[str, tuple[float, float]] = {
    "Observ_jud": (7.5, 0.02),
    "Re_count_si": (16.0, 0.0),
    "Re_count_no": (20.0, 0.0),
    "Count_req_by": (6.0, 0.02),
    "On_behalf_of": (6.0, 0.02),
    "Firma_J1": (4.5, 0.02),
    "Firma_J2": (4.5, 0.02),
    "Firma_J3": (4.5, 0.02),
    "Firma_J4": (4.5, 0.02),
    "Firma_J5": (4.5, 0.02),
    "Firma_J6": (4.5, 0.02),
    "Firma_cc_J1": (7.5, 0.02),
    "Firma_cc_J2": (7.5, 0.02),
    "Firma_cc_J3": (7.5, 0.02),
    "Firma_cc_J4": (7.5, 0.02),
    "Firma_cc_J5": (7.5, 0.02),
    "Firma_cc_J6": (7.5, 0.02),
}


def load_trocr_resources(
    model_name: str = DEFAULT_TROCR_MODEL,
    vocab: str = DEFAULT_VOCAB,
    device: str | torch.device | None = None,
) -> tuple[torch.device, TrOCRProcessor, VisionEncoderDecoderModel, dict[str, list[int]]]:
    """Inicializa una única vez en memoria el modelo TrOCR, su procesador y el mapa de tokens.

    Args:
        model_name (str, opcional): Identificador del modelo en HuggingFace.
            Por defecto es "microsoft/trocr-base-handwritten".
        vocab (str, opcional): Caracteres objetivo permitidos durante la decodificación.
            Por defecto es "0123456789.*#-/\\".
        device (str | torch.device | None, opcional): Dispositivo de cómputo ('cuda' o 'cpu').
            Si es None, selecciona GPU automáticamente si está disponible.

    Returns:
        tuple: Cuádrupla (device, processor, model, vocab_token_ids) lista para inferencia.
    """
    # Detección automática del hardware disponible (CUDA o CPU)
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # Inicialización desacoplada con RobertaTokenizer para evitar fallos de conversión fast-tokenizer
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    processor = TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)

    # Carga del modelo en modo evaluación (desactiva dropout y gradientes)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(target_device).eval()

    # Mapeo de cada carácter objetivo hacia todos sus posibles IDs BPE (con o sin espacio previo 'Ġ')
    vocab_token_ids: dict[str, list[int]] = {char: [] for char in vocab}
    for token_str, token_id in processor.tokenizer.get_vocab().items():
        decoded = processor.tokenizer.convert_tokens_to_string([token_str]).strip()
        if decoded in vocab_token_ids:
            vocab_token_ids[decoded].append(token_id)

    return target_device, processor, model, vocab_token_ids


def load_easyocr_reader(
    languages: list[str] | None = None,
    gpu: bool | None = None,
) -> easyocr.Reader:
    """Inicializa una única vez en memoria el lector EasyOCR para texto impreso.

    Args:
        languages (list[str] | None, opcional): Códigos de idioma soportados. Por defecto ['es'].
        gpu (bool | None, opcional): Habilita aceleración por GPU. Si es None, autodetecta CUDA.

    Returns:
        easyocr.Reader: Instancia inicializada del detector/lector EasyOCR.
    """
    langs = languages or ["es"]
    use_gpu = torch.cuda.is_available() if gpu is None else gpu
    return easyocr.Reader(langs, gpu=use_gpu)


def evaluate_omr_cell(
    binary_crop: np.ndarray,
    threshold_pct: float = 5.0,
    inner_margin_pct: float = 0.0,
) -> tuple[bool, float]:
    """Evalúa si una casilla binarizada está marcada calculando la densidad de píxeles de tinta.

    Args:
        binary_crop (np.ndarray): Matriz binarizada (fondo blanco = 255, tinta negra = 0).
        threshold_pct (float, opcional): Porcentaje mínimo de tinta (0 a 100) para marcar True.
            Por defecto es 5.0.
        inner_margin_pct (float, opcional): Fracción de recorte perimetral (0.0 a 0.49) para
            descartar las líneas impresas del recuadro. Por defecto es 0.0.

    Raises:
        ValueError: Si el margen interior consume toda el área de la matriz.

    Returns:
        tuple[bool, float]: Estado booleano de marca detectada y porcentaje de llenado (2 decimales).
    """
    # Conversión preventiva a 2D si el recorte ingresa en 3 canales
    if binary_crop.ndim == 3:
        gray_crop = cv2.cvtColor(binary_crop, cv2.COLOR_RGB2GRAY)
    else:
        gray_crop = binary_crop

    # Inversión binaria: OpenCV requiere tinta en blanco (255) y fondo en negro (0) para countNonZero
    white_ink_mask = cv2.bitwise_not(gray_crop)

    # Recorte de seguridad interior para aislar el núcleo de la casilla
    height, width = white_ink_mask.shape
    margin_y = int(height * inner_margin_pct)
    margin_x = int(width * inner_margin_pct)
    cell_core = white_ink_mask[margin_y : height - margin_y, margin_x : width - margin_x]

    # Cálculo de área efectiva y porcentaje de píxeles activos
    total_area = cell_core.shape[0] * cell_core.shape[1]
    if total_area <= 0:
        raise ValueError("Error de dimensiones: El margen interior consumió toda el área de la matriz.")

    ink_pixels = cv2.countNonZero(cell_core)
    fill_percentage = (ink_pixels / total_area) * 100.0
    is_marked = fill_percentage >= threshold_pct

    return is_marked, round(fill_percentage, 2)


def predict_handwritten_cell(
    img_crop: np.ndarray,
    label: str,
    device: torch.device | str,
    vocab_token_ids: dict[str, list[int]],
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
    top_k: int = 5,
    blank_threshold_pct: float = 1.0,
) -> list[dict[str, str | float]]:
    """Clasifica un carácter manuscrito individual restringido al vocabulario objetivo mediante TrOCR.

    Args:
        img_crop (np.ndarray): Recorte de imagen de una sola casilla numérica.
        label (str): Nombre identificador de la casilla (ej. 'Votos_A_1').
        device (torch.device | str): Dispositivo de cómputo ('cuda' o 'cpu').
        vocab_token_ids (dict[str, list[int]]): Diccionario de caracteres a IDs BPE.
        processor (TrOCRProcessor): Preprocesador de imagen y texto de TrOCR.
        model (VisionEncoderDecoderModel): Modelo TrOCR cargado en modo evaluación.
        top_k (int, opcional): Número de mejores predicciones a retornar. Por defecto es 5.
        blank_threshold_pct (float, opcional): Porcentaje mínimo de tinta para evitar clasificar
            la celda directamente como 'EN_BLANCO'. Por defecto es 1.0.

    Returns:
        list[dict[str, str | float]]: Lista ordenada de predicciones con esquema fijo:
        [{"field": label, "prediction": char, "score": float}].
    """
    # Filtro OMR previo: omite inferencia neuronal si la celda está vacía
    is_marked, _ = evaluate_omr_cell(img_crop, threshold_pct=blank_threshold_pct)
    if not is_marked:
        return [{"field": label, "prediction": "EN_BLANCO", "score": 1.0}]

    # TrOCR exige entrada RGB de 3 canales
    if img_crop.ndim == 2:
        crop_rgb = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2RGB)
    else:
        crop_rgb = img_crop

    # Preparación de tensores y token inicial forzado para evaluar únicamente el primer carácter
    pixel_values = processor(images=crop_rgb, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = torch.tensor(
        [[model.generation_config.decoder_start_token_id]], device=device
    )

    # Paso forward sin cálculo de gradientes y conversión de logits a probabilidades softmax
    with torch.no_grad():
        logits = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids).logits
        probs = F.softmax(logits[0, -1], dim=-1)

    # Agregación de probabilidades por carácter objetivo y ordenamiento descendente
    scores = {
        char: sum(probs[token_id].item() for token_id in token_ids)
        for char, token_ids in vocab_token_ids.items()
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    return [{"field": label, "prediction": char, "score": score} for char, score in ranked]


def read_text_region(
    img_input: np.ndarray | str,
    reader: easyocr.Reader,
    sort_axis: str = "y",
) -> list[dict[str, str | float]]:
    """Extrae texto impreso de un recorte de imagen usando EasyOCR.

    Args:
        img_input (np.ndarray | str): Matriz de imagen del recorte o ruta de archivo.
        reader (easyocr.Reader): Instancia inicializada de EasyOCR.
        sort_axis (str, opcional): Eje de ordenamiento de cajas detectadas: 'y' (vertical,
            arriba hacia abajo) o 'x' (horizontal, izquierda a derecha). Por defecto es 'y'.

    Returns:
        list[dict[str, str | float]]: Lista de bloques detectados [{"text": str, "score": float}].
    """
    # Detección y reconocimiento de texto en una sola pasada
    raw_results = reader.readtext(img_input)
    if not raw_results:
        return [{"text": "Ilegible", "score": 0.0}]

    # Ordenamiento por coordenada X (0) para líneas únicas o Y (1) para bloques multilínea
    axis_idx = 0 if sort_axis.lower() == "x" else 1
    raw_results.sort(key=lambda item: item[0][0][axis_idx])

    return [{"text": str(text), "score": float(score)} for _, text, score in raw_results]


def _merge_single_line_ocr(
    ocr_items: list[dict[str, str | float]],
    strip_form_code: bool = False,
) -> tuple[str, float]:
    """Une fragmentos horizontales de EasyOCR en una sola cadena y conserva la confianza mínima.

    Args:
        ocr_items (list[dict[str, str | float]]): Bloques ordenados por eje X.
        strip_form_code (bool, opcional): Si es True, elimina guiones y posibles letras 'X'
            laterales del código X-#-#-#-# X. Por defecto es False.

    Returns:
        tuple[str, float]: Tupla (texto_limpio, confianza_minima).
    """
    if not ocr_items or ocr_items[0]["text"] == "Ilegible":
        return "Ilegible", 0.0

    # Descarta cajas aisladas que solo correspondan a la 'X' lateral del código de seguridad
    filtered = (
        [item for item in ocr_items if str(item["text"]).strip().upper() != "X"]
        if strip_form_code
        else ocr_items
    )
    if not filtered:
        return "Ilegible", 0.0

    merged_text = "".join(str(item["text"]) for item in filtered).replace(" ", "")
    if strip_form_code:
        merged_text = merged_text.replace("-", "").upper().replace("X", "")

    min_score = min(float(item["score"]) for item in filtered)
    return (merged_text if merged_text else "Ilegible"), min_score


def extract_e14_ocr_data(
    rois_list: list[dict[str, np.ndarray]],
    reader: easyocr.Reader,
    device: torch.device | str,
    vocab_token_ids: dict[str, list[int]],
    processor: TrOCRProcessor,
    model: VisionEncoderDecoderModel,
) -> tuple[dict[str, Any] | None, dict[str, str | None]]:
    """Orquesta EasyOCR, TrOCR y OMR sobre los 4 bloques ROI de un formulario E-14.

    Args:
        rois_list (list[dict[str, np.ndarray]]): Lista con los 4 diccionarios ROI
            [comun_p1, comun_p2, celdas_p1, omr_p2].
        reader (easyocr.Reader): Lector EasyOCR cargado en memoria.
        device (torch.device | str): Hardware de cómputo ('cuda' o 'cpu').
        vocab_token_ids (dict[str, list[int]]): Mapa de tokens BPE del vocabulario.
        processor (TrOCRProcessor): Preprocesador de imágenes de TrOCR.
        model (VisionEncoderDecoderModel): Modelo TrOCR en modo evaluación.

    Returns:
        tuple[dict[str, Any] | None, dict[str, str | None]]: Diccionario consolidado de campos
        y diccionario DIVIPOLE de la Página 1. Retorna (None, {}) si ocurre una excepción.
    """
    try:
        # 1. Lectura y comparación de metadatos DIVIPOLE entre Página 1 y Página 2
        divipole_p1 = parse_divipole_text(read_text_region(rois_list[0]["Divipole"], reader))
        divipole_p2 = parse_divipole_text(read_text_region(rois_list[1]["Divipole"], reader))
        review_flag = {"manual": divipole_p1 != divipole_p2}

        # 2. Lectura horizontal de códigos de barras de Página 1 y Página 2
        b1_text, b1_score = _merge_single_line_ocr(
            read_text_region(rois_list[0]["Barras"], reader, sort_axis="x")
        )
        b2_text, b2_score = _merge_single_line_ocr(
            read_text_region(rois_list[1]["Barras"], reader, sort_axis="x")
        )
        barcodes = {
            "Barras_1": b1_text,
            "s_barras_1": b1_score,
            "Barras_2": b2_text,
            "s_barras_2": b2_score,
        }

        # 3. Lectura cruzada horizontal del código X-#-#-#-# X en Página 1 y Página 2
        c1_text, c1_score = _merge_single_line_ocr(
            read_text_region(rois_list[0]["Codigo"], reader, sort_axis="x"),
            strip_form_code=True,
        )
        c2_text, c2_score = _merge_single_line_ocr(
            read_text_region(rois_list[1]["Codigo"], reader, sort_axis="x"),
            strip_form_code=True,
        )
        form_code = {
            "Codigo": c1_text,
            "s_codigo": c1_score,
            "Codigo_2": c2_text,
            "s_codigo_2": c2_score,
        }

        # 4. Inferencia TrOCR sobre las 27 casillas manuscritas de votación (Página 1)
        handwritten_cells: dict[str, str | float] = {}
        for label, crop in rois_list[2].items():
            top_pred = predict_handwritten_cell(
                crop, label, device, vocab_token_ids, processor, model, top_k=1
            )[0]
            field_name = str(top_pred["field"])
            handwritten_cells[field_name] = str(top_pred["prediction"])
            handwritten_cells[f"s_{field_name}"] = float(top_pred["score"])

        # 5. Evaluación OMR calibrada por zona sobre firmas, constancias y recuento (Página 2)
        omr_cells: dict[str, tuple[bool, float]] = {}
        for label, crop in rois_list[3].items():
            threshold_pct, inner_margin_pct = OMR_ZONE_THRESHOLDS.get(label, (5.0, 0.0))
            omr_cells[label] = evaluate_omr_cell(
                crop, threshold_pct=threshold_pct, inner_margin_pct=inner_margin_pct
            )

        # Consolidación de todos los diccionarios en una sola estructura plana
        consolidated_data = (
            review_flag
            | form_code
            | barcodes
            | divipole_p1
            | handwritten_cells
            | omr_cells
        )
        return consolidated_data, divipole_p1

    except Exception as err:
        print(f"[ERROR OCR] Fallo procesando regiones del formulario: {err}")
        return None, {}