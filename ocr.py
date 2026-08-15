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


def es_imagen_en_blanco(img_crop, umbral_oscuridad=127, min_proporcion_tinta=0.01):
    """
    REEMPLAZAR por def evaluar_casilla_omr( Determina si la imagen no tiene ningún carácter 
    dibujado, contando qué proporción de píxeles son "oscuros" (tinta) frente al total.

    Args:
        img_crop (numpy_array): Recorte de imagen a analizar
        umbral_oscuridad (int, optional): valor de gris (0-255) por debajo del cual un
            píxel se considera "tinta" en vez de fondo. Defaults to 127.
        min_proporcion_tinta (float, optional): proporción mínima de píxeles de tinta para
            considerar que SÍ hay un carácter. [Bajar] si los trazos son muy
            finos, o [Subir] si detecta ruido/manchas como si fuera texto. Defaults to 0.01.

    Returns:
        bool: _description_
    """
    #Se convierte a escala de grises si viene en 3 canales (RGB/BGR)
    if img_crop.ndim == 3:
        gris = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
    else:
        gris = img_crop

    #Conteo de píxeles oscuros (tinta)mediante OpenCV
    # THRESH_BINARY_INV asigna 255 a píxeles < umbral_oscuridad y 0 a los claros 
    _, mascara_tinta = cv2.threshold(gris, umbral_oscuridad, 255, cv2.THRESH_BINARY_INV)
    #countNonZero Necesitaba Fondo negro, trazos blancos, hecho en la linea anterior
    pixeles_tinta = cv2.countNonZero(mascara_tinta)

    #Calcula la proporción sobre el total de píxeles
    proporcion = pixeles_tinta / gris.size

    return proporcion < min_proporcion_tinta
# ---------------------------------------------------------------------------
# Función principal
# ---------------------------------------------------------------------------

def bloque_ocr(img_crop, label, device, vocab_token_ids, processor, model, top_k=5):
    """
    Clasifica el carácter manuscrito de la imagen dentro del vocabulario 0123456789.*#-/\\ y 
    devuelve las top_k predicciones como lista de dicts {"label": ..., "score": ...}, igual 
    que la salida nativa de un pipeline de clasificación.
        
    Args:
        img_crop (numpy_array): Direccion de imagen a digitalizar
        label (str):          Etiqueta de la celda que se va a procesar
        device (str):           Hardware con el que se procesa "cpu" o "cuda"
        vocab_token_ids (dict): Diccionario con keys de vocabulario objetivo
        processor (_type_):     modulo que preprocesa la img para el modelo
        model (_type_):         Modelo OCR en modo eval() - Inferencia
        top_k (int, optional): Cantidad de resultados requeridos. Defaults to 5.

    Returns:
        dict: Diccionario con las *top_k* mejores predicciones y su probabilidad de mayor a menor
    """
    #Se asegura formato de 3 canales (RGB) requerido por TrOCRProcessor
    if img_crop.ndim == 2:
        #De 1 canal (alto, ancho) a 3 canales (alto, ancho, 3)
        crop_rgb = cv2.cvtColor(img_crop, cv2.COLOR_GRAY2RGB)
    else:
        crop_rgb = img_crop

    #Verificación de imagen en blanco directamente sobre el ndarray
    # (Suponiendo binarización THRESH_BINARY_INV: Fondo=0, Tinta=255)
    if es_imagen_en_blanco(crop_rgb):
        return [{f"{label}": "EN_BLANCO", "score": 1.0}]
    
    #Se convierten los pixeles a tensores
    pixel_values = processor(images=crop_rgb, return_tensors="pt").pixel_values.to(device)

    #Se forza el token de inicio (decoder_start_token_id) para evaluar únicamente la predicción del primer carácter.
    decoder_input_ids = torch.tensor([[model.generation_config.decoder_start_token_id]], device=device)

    #Se procesa la imagen en el modelo y devuelve puntuaciones de probabilidad para los tokens
    with torch.no_grad():
        logits = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids).logits
        probs = F.softmax(logits[0, -1], dim=-1)

    #Se hace una sumatoria de todas las probabilidades de todos los tokens asociados cada vocablo objetivo
    scores = {c: sum(probs[i].item() for i in ids) for c, ids in vocab_token_ids.items()}
    #Organiza el diccionario de predicciones resultante de mayor a menor 
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    return [{f"{label}": c, f"s_{label}": s} for c, s in ranked]


# ---------------------------------------------------------------------------# ---------------------------------------------------------------------------
# Modelo OCR para leer Divipole
# ---------------------------------------------------------------------------# ---------------------------------------------------------------------------
# En Colab, instalar primero:
# !pip install -q easyocr
def ocr_documento(image_path, reader):
    """
    Recibe la RUTA de la imagen completa del documento (sin necesidad de partirla por líneas) 
    y devuelve el texto detectado, ordenado de arriba hacia abajo.

    Args:
        image_path (numpy_array): Recorte de documento que se va a leer
        reader (_type_): Modelo cargado de easyocr
    Returns:
        _type_: _description_
    """

    #Devuelve una lista de tuplas con la estructura [(bbox, texto, confianza), ...]
    # bbox: Lista de 4 puntos con coordenadas [X, Y] de las esquinas del cuadro [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    resultados = reader.readtext(image_path)  # detección + reconocimiento en un solo paso

    # Cada resultado es (bbox, texto, confianza). Ordenamos por la posición
    # vertical (eje Y) de cada línea para respetar el orden de lectura.
    resultados.sort(key=lambda r: r[0][0][1])

    # Omitimos bbox en el for por que no es necesario
    return [{f"texto": texto, "score": score} for _, texto, score in resultados]

# ---------------------------------------------------------------------------# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Detección de imagen en blanco (sin ningún trazo) Optical Mark Recognition
# ---------------------------------------------------------------------------

def eval_cell_omr(recorte_binario, umbral_porcentaje=5.0, margen_interior_pct=0.0):
    """
    Evalúa si una casilla binarizada está marcada basándose en la densidad de píxeles.

    Args:
        recorte_binario (numpy_array): Matriz de imagen ya binarizada
        umbral_porcentaje (float, optional):  % mínimo de píxeles de tinta para considerarla marcada.
            (Marcas sutiles como una 'X' rápida pueden ocupar solo el 4-8% del área). Defaults to 5.0.
        margen_interior_pct (float, optional): % del borde a ignorar para no contabilizar las líneas 
            de la caja. Defaults to 0.0.
    Returns:
        bool, float : Devuelve si el recorte de la img es mayor o igual al umbral y el % de llenado.
    """
        
    # Asume que el input viene de cv2.THRESH_BINARY (fondo blanco, tinta negra) y se invierte
    tinta_blanca = cv2.bitwise_not(recorte_binario)
    
    #Se calcula las dimensiones para el recorte interior de seguridad
    alto, ancho = tinta_blanca.shape
    margen_y = int(alto * margen_interior_pct)
    margen_x = int(ancho * margen_interior_pct)
    
    #Se extrae el núcleo de la casilla (excluyendo bordes)
    nucleo_casilla = tinta_blanca[margen_y : alto - margen_y, margen_x : ancho - margen_x]
    #view_lite(nucleo_casilla) #Prueba de visualizacion
    
    #Se hace el cálculo algorítmico de densidad
    area_total = nucleo_casilla.shape[0] * nucleo_casilla.shape[1]
    
    if area_total == 0:
        raise ValueError("Error de dimensiones: El margen interior consumió toda el área de la matriz.")
        
    pixeles_tinta = cv2.countNonZero(nucleo_casilla)
    porcentaje_llenado = (pixeles_tinta / area_total) * 100.0
    
    #Evaluación booleana
    estado = porcentaje_llenado >= umbral_porcentaje
    
    return estado, round(porcentaje_llenado, 2)



