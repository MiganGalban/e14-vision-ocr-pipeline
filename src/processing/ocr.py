### HWR_Colab_2.ipynb guardado en Drive
import easyocr
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import numpy as np
import cv2


#---------------------------------------------------------------------------#---------------------------------------------------------------------------
# Configuración: Modelo OCR y vocabulario objetivo (Numeros escritos a mano)
#---------------------------------------------------------------------------#---------------------------------------------------------------------------
DEFAULT_TROCR_MODEL = "microsoft/trocr-base-handwritten"
DEFAULT_VOCAB = "0123456789.*#-/\\"
def load_trocr_resources(   model_name: str = DEFAULT_TROCR_MODEL,
                            vocab: str = DEFAULT_VOCAB, 
                            device: str | torch.device | None = None,
                        ) -> tuple[torch.device, TrOCRProcessor, VisionEncoderDecoderModel, dict[str, list[int]]]:
    """
    Inicializa una única vez en memoria el modelo TrOCR, su procesador y el mapa de tokens.

    Args:
        model_name (str, opcional): Identificador del modelo en HuggingFace.
            Por defecto es "microsoft/trocr-base-handwritten".
        vocab (str, opcional): Cadena de caracteres permitidos para restringir la inferencia.
            Por defecto es "0123456789.*#-/\\".
        device (str | torch.device | None, opcional): Dispositivo de cómputo ('cuda' o 'cpu').
            Si es None, detecta automáticamente disponibilidad de GPU.

    Returns:
        tuple: Tupla con (device, processor, model, vocab_token_ids).
    """
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(target_device).eval()

    vocab_token_ids: dict[str, list[int]] = {char: [] for char in vocab}
    for token_str, token_id in processor.tokenizer.get_vocab().items():
        decoded = processor.tokenizer.convert_tokens_to_string([token_str]).strip()
        if decoded in vocab_token_ids:
            vocab_token_ids[decoded].append(token_id)

    return target_device, processor, model, vocab_token_ids

def load_easyocr_reader(    languages: list[str] | None = None, 
                            gpu: bool | None = None,
                        ) -> easyocr.Reader:
    
    """
    Inicializa una única vez en memoria el lector EasyOCR.

    Args:
        languages (list[str] | None, opcional): Lista de códigos de idioma.
            Por defecto es ['es'].
        gpu (bool | None, opcional): Habilita aceleración por GPU. Si es None,
            se activa automáticamente si CUDA está disponible.

    Returns:
        easyocr.Reader: Instancia cargada del lector OCR.
    """
    langs = languages or ["es"]
    use_gpu = torch.cuda.is_available() if gpu is None else gpu
    return easyocr.Reader(langs, gpu=use_gpu)




# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------


def predict_handwritten_cell(   img_crop: np.ndarray,   label: str, 
                                device: torch.device | str, 
                                vocab_token_ids: dict[str, list[int]], 
                                processor: TrOCRProcessor,
                                model: VisionEncoderDecoderModel, top_k: int = 5,     
                                blank_threshold_pct: float = 1.0,
                            ) -> list[dict[str, str | float]]:
    """
    Clasifica el carácter manuscrito dentro del vocabulario objetivo 0123456789.*#-/\\ mediante TrOCR.
    Verifica primero mediante densidad OMR si la celda está vacía. Si contiene trazos,
    ejecuta un paso de decodificación forzada y suma las probabilidades BPE por carácter.
    Retorna las top_k predicciones como lista de Diccionarios.

        img_crop (np.ndarray):  Recorte de imagen de la celda a evaluar.
        label (str):    Etiqueta identificadora de la celda (ej. 'Votos_A_1').
        device (torch.device | str):    Dispositivo de cómputo ('cuda' o 'cpu').
        vocab_token_ids (dict[str, list[int]]):     Mapa de caracteres a IDs de tokens BPE.
        processor (TrOCRProcessor):     Procesador de imágenes y texto de TrOCR.
        model (VisionEncoderDecoderModel):  Modelo TrOCR cargado en modo evaluación.
        top_k (int, opcional):  Cantidad de predicciones principales a retornar. Por defecto es 5.
        blank_threshold_pct (float, opcional):  Porcentaje mínimo de tinta para no declarar
            la casilla como 'EN_BLANCO'. Por defecto es 1.0 (1%).

    Returns:
        list[dict[str, str | float]]:   Lista de diccionarios ordenada por probabilidad 
            descendente con esquema fijo: 
            [{"field": label, "prediction": char, "score": prob}].
    """
    #Verificación de imagen en blanco directamente sobre el ndarray

    is_marked, _ = evaluate_omr_cell(img_crop, threshold_pct=blank_threshold_pct)
    if not is_marked:
        return [{"field": label, "prediction": "EN_BLANCO", "score": 1.0}]

    #Se asegura formato de 3 canales (RGB) requerido por TrOCRProcessor
    if img_crop.ndim == 2:
        #De 1 canal (alto, ancho) a 3 canales (alto, ancho, 3)
        crop_rgb = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2RGB)
    else:
        crop_rgb = img_crop

    
    
    #Se convierten los pixeles a tensores
    pixel_values = processor(images=crop_rgb, return_tensors="pt").pixel_values.to(device)

    #Se forza el token de inicio (decoder_start_token_id) para evaluar únicamente la predicción del primer carácter.
    decoder_input_ids = torch.tensor( [[model.generation_config.decoder_start_token_id]], 
                                        device=device )
    #Se procesa la imagen en el modelo y devuelve puntuaciones de probabilidad para los tokens
    with torch.no_grad():
        logits = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids).logits
        probs = F.softmax(logits[0, -1], dim=-1)

    #Se hace una sumatoria de todas las probabilidades de todos los tokens asociados cada vocablo objetivo
    scores = {
                char: sum(probs[token_id].item() for token_id in token_ids)
                for char, token_ids in vocab_token_ids.items()
            }
    
    #Organiza el diccionario de predicciones resultante de mayor a menor 
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    return [{"field": label, "prediction": char, "score": score} for char, score in ranked]


# ---------------------------------------------------------------------------# ---------------------------------------------------------------------------
# Modelo OCR para leer Divipole
# ---------------------------------------------------------------------------# ---------------------------------------------------------------------------


def read_text_region(   img_input: np.ndarray | str,  
                        reader: easyocr.Reader,
                    ) -> list[dict[str, str | float]]:
    """
    Extrae texto impreso de un recorte de documento o ruta de imagen usando EasyOCR.
    Ordena los bloques detectados verticalmente (de arriba hacia abajo) según su coordenada Y.

    Args:
        img_input (np.ndarray | str): Matriz de imagen del recorte o ruta del archivo.
        reader (easyocr.Reader): Instancia inicializada de EasyOCR.

    Returns:
        list[dict[str, str | float]]: Lista de diccionarios con estructura [{"text": str, "score": float}].
    """

    #Devuelve una lista de tuplas con la estructura [(bbox, texto, confianza), ...]
    # bbox: Lista de 4 puntos con coordenadas [X, Y] de las esquinas del cuadro [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    raw_results = reader.readtext(img_input)  # detección + reconocimiento en un solo paso
    if not raw_results:
        return [{"text": "Ilegible", "score": 0.0}]
    # Cada resultado es (bbox, texto, confianza). Ordenamos por la posición
    # vertical (eje Y) de cada línea para respetar el orden de lectura.
    raw_results.sort(key=lambda item: item[0][0][1])

    # Omitimos bbox en el for por que no es necesario
    return [{"text": str(text), "score": float(score)} for _, text, score in raw_results] 

# ---------------------------------------------------------------------------
# Detección de imagen en blanco (sin ningún trazo) Optical Mark Recognition
# ---------------------------------------------------------------------------

def evaluate_omr_cell(  binary_crop: np.ndarray, threshold_pct: float = 5.0,
                        inner_margin_pct: float = 0.0, 
                    ) -> tuple[bool, float]:
    """
    Evalúa si una casilla binarizada está marcada calculando la densidad de píxeles de tinta.
    Asume que la imagen de entrada proviene de una binarización estándar (cv2.THRESH_BINARY:
    fondo blanco = 255, tinta negra = 0) e invierte la máscara para contabilizar los trazos.

    Args:
        binary_crop (np.ndarray): Matriz de imagen binarizada.
        threshold_pct (float, opcional): Porcentaje mínimo de píxeles de tinta (0 a 100)
            para considerar la casilla como marcada/diligenciada. Por defecto es 5.0.
        inner_margin_pct (float, opcional): Fracción del borde (0.0 a 0.49) que se recorta
            hacia el interior para ignorar las líneas del recuadro. Por defecto es 0.0.

    Raises:
        ValueError: Si el margen interior consume la totalidad del área de la matriz.

    Returns:
        tuple[bool, float]: Estado booleano (True si supera el umbral) y porcentaje de llenado
        redondeado a 2 decimales.
    """
        
    # Asume que el input viene de cv2.THRESH_BINARY (fondo blanco, tinta negra) y se invierte
    if binary_crop.ndim == 3:
        gray_crop = cv2.cvtColor(binary_crop, cv2.COLOR_RGB2GRAY)
    else:
        gray_crop = binary_crop

    white_ink_mask = cv2.bitwise_not(gray_crop)
    
    #Se calcula las dimensiones para el recorte interior de seguridad
    height, width = white_ink_mask.shape
    margin_y = int(height * inner_margin_pct)
    margin_x = int(width * inner_margin_pct)
    
    #Se extrae el núcleo de la casilla (excluyendo bordes)
    cell_core = white_ink_mask[margin_y : height - margin_y, margin_x : width - margin_x]
    #view_lite(nucleo_casilla) #Prueba de visualizacion
    
    #Se hace el cálculo algorítmico de densidad
    total_area = cell_core.shape[0] * cell_core.shape[1]
    
    if total_area <= 0:
        raise ValueError("Error de dimensiones: El margen interior consumió toda el área de la matriz.")
        
    ink_pixels = cv2.countNonZero(cell_core)
    fill_percentage = (ink_pixels / total_area) * 100.0
    
    #Evaluación booleana
    is_marked = fill_percentage >= threshold_pct

    return is_marked, round(fill_percentage, 2)