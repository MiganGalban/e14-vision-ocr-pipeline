import cv2
import numpy as np
from pdf2image import convert_from_path
import matplotlib.pyplot as plt

from view import view_all_perc

def pdf_to_img(ruta_pdf):
    """
    Convierte de PDF a Imagen B/N alineada, usando la funcion "align_pag"
    Args:
        ruta_pdf (_type_): Ruta del PDF a convertir
    Returns:
        List : Retorna una lista con la cantidad de paginas que tenia el PDF en 
        array de numpy Ej: resultados[0] es una pagina a B/N alineada
    """
    paginas_pil = convert_from_path(ruta_pdf, dpi=300)
    # Toma el PDF lo pasa a imagenes de 300 DPI q es lo minimo y optimo para OCR
    resultados = []
    for i, pagina in enumerate(paginas_pil):
        img_rgb = np.array(pagina)
        #view_all_perc(img_rgb, 24, 36) #Prueba visualizacion antes de convertir
        try:
            img_alineada = align_pag(img_rgb)
            resultados.append(img_alineada)
            print(f"Página {i+1}: Registro completado.")
        except ValueError as e:
            #Manejo de fallo en caso de que hayan problemas en la func align_pag
            print(f"Página {i+1}: Error -> {e}")
            resultados.append(None) 
            

    return resultados

def align_pag(img_rgb, margen_w=0.15, margen_h=0.04, tolerancia_aspecto=0.1):
    """
    Toma los cuadros de referencia de las esquinas para enderezar, nivelar y 
    estirar la página inclinada o deformada
    Args:
        img_rgb (numpy_array): e14 escaneado
        
        margen_w (float, optional): El porcentaje de pixeles a lo ancho que se 
        toman para buscar los cuadros guias de las esquinas. Defaults to 0.15.
        
        margen_h (float, optional): El porcentaje de pixeles que se toma en 
        altura para buscar los cuadros guias de las esquinas. Defaults to 0.04.
        
        tolerancia_aspecto (float, optional): Tolerancia de variacion del cuadro
        que se esta buscando. Defaults to 0.1.
    
    Returns:
        List: Lista de numpy_array de las imagenes alineadas, ajustadas y 
        binarizadas (B/N). Cada pagina es un numpy_array en la lista.
    """
    # Conversión para procesamiento OpenCV
    img_gris = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    alto, ancho = img_gris.shape

    #Margenes en pixeles
    m_x, m_y = int(ancho * margen_w), int(alto * margen_h)
    
    # Zonas de búsqueda en orden estricto: TL, TR, BL, BR
    # Diccionario de zonas de interes recortados ROI y offset
    # "key": imagen[y_inicio:y_fin, x_inicio:x_fin], (offset_x, offset_y) 
    zonas = {
        "Top-Left":   (img_gris[0:m_y, 0:m_x], (0, 0)),
        "Top-Right":  (img_gris[0:m_y, ancho - m_x:ancho], (ancho - m_x, 0)),
        "Bottom-Left":  (img_gris[alto - m_y:alto, 0:m_x], (0, alto - m_y)),
        "Bottom-Right": (img_gris[alto - m_y:alto, ancho - m_x:ancho], (ancho - m_x, alto - m_y))
    }

    puntos_origen = []
    
    for nombre, (roi, offset) in zonas.items():
        #cv2.THRESH_BINARY_INV: Invierte los colores, OpenCV exige que los objetos a detectar sean blancos
        #cv2.THRESH_OTSU: Calcula automáticamente el umbral de separación óptimo entre fondo y figura
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #La función devuelve una tupla (valor_de_umbral, imagen_binarizada).

        #cv2.RETR_EXTERNAL: Extrae los contornos exteriores más externos
        #cv2.CHAIN_APPROX_SIMPLE: Comprime lineas rectas a dos vertices 
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #La función devuelve una lista con los contornos de cada cuadro que encontro

        marca_encontrada = False
        area_roi = roi.shape[0] * roi.shape[1]
        
        for cnt in contornos:
            #Calcula el area de los contornos y aplica una restriccion para descartar falsos positivos
            area = cv2.contourArea(cnt)
            if not (0.002 * area_roi < area < 0.3 * area_roi): continue

            #Calcula el perimetro y simplifica la forma a vertices
            perimetro = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimetro, True)

            if len(approx) == 4:
                #Calcula la forma delimitadora para luego probar si es un cuadro 
                #(teniendo en cuenta la tolerancia)
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h

                if (1.0 - tolerancia_aspecto) <= aspect_ratio <= (1.0 + tolerancia_aspecto):
                    #Se obtiene el centro del cuadrado
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"]) + offset[0]
                        cy = int(M["m01"] / M["m00"]) + offset[1]
                        puntos_origen.append([cx, cy])
                        marca_encontrada = True
                        break

        if not marca_encontrada:
            raise ValueError(f"Fallo geométrico: Marca no detectada en {nombre}.")

    pts_origen = np.array(puntos_origen, dtype=np.float32)

    # Mapeo a rectangulo perfecto (e14) respetando dimensiones originales
    pts_destino = np.array([
        [0, 0],
        [ancho - 1, 0],
        [0, alto - 1],
        [ancho - 1, alto - 1]
    ], dtype=np.float32)

    #Calcula la matriz de conversion para ajustar la imagen a un rectangulo
    matriz_perspectiva = cv2.getPerspectiveTransform(pts_origen, pts_destino)

    #Aplica la matriz para convertir la imagen a un rectangulo
    img_alineada = cv2.warpPerspective(img_rgb, matriz_perspectiva, (ancho, alto))

    view_all_perc(img_alineada) #Prueba de visualizacion de Img

    #Se toma la imagen alineada y recortada en los cuadros para aplicar filtro B/N
    img_bin_adp = bin_adapt(img_alineada, block_size=81, C=40) 

    return img_bin_adp




def bin_adapt(crop_img, block_size=81, C=40):
    """
    Aplica binarización adaptativa. Util para imagenes con multiples elementos.

    Args:
        crop_img (numpy_array): Imagen 
        block_size (int, optional): Tamaño de píxeles a analizar(debe ser impar). 
        Un valor mayor evalúa un área más amplia, útil para variaciones de iluminación 
        grandes. Defaults to 81.
        C (int, optional): Constante sustraída de la media calculada. Ajusta la 
        sensibilidad de la segmentación. Defaults to 40.

    Returns:
        resultado_visual (numpy_array): Retorna imagen binarizada (Fondo blanco,
        trazos negros )
    """

    # Validación de matriz bidimensional (Escala de Grises)

    if len(crop_img.shape) == 3:
        gris = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gris = crop_img
        
    # Aplicación del umbral adaptativo. Se hicieron pruebas y se observo que un 
    # punto optimo para los e14 era block_size=81, C=40
    resultado_visual = cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C
    )

    return resultado_visual
    