from pathlib import Path
import cv2
import numpy as np
from pdf2image import convert_from_path
import matplotlib.pyplot as plt

from src.processing.view import plot_aligned_grid

#from view import view_all_perc, view_lite

def pdf_to_images(  pdf_path: str | Path, dpi: int = 300, debug: bool = False,
                    ) -> list[np.ndarray | None]:
    """
    Rasteriza un archivo PDF a imágenes de 300 DPI y alinea cada página.

    Args:
        pdf_path (str | Path): Ruta del archivo PDF a procesar.
        dpi (int, opcional): Resolución de conversión en puntos por pulgada. Por defecto es 300.
        debug (bool, opcional): Activa visualizaciones de depuración durante la alineación.
            Por defecto es False.

    Returns:
        list[np.ndarray | None]: Lista con las matrices binarizadas y alineadas por página
        (o None en la posición de la página si falló la alineación geométrica).
    """
    pil_pages = convert_from_path(str(pdf_path), dpi=dpi)
    aligned_pages: list[np.ndarray | None] = []

    # Toma el PDF lo pasa a imagenes de 300 DPI q es lo minimo y optimo para OCR
    for idx, page in enumerate(pil_pages):
        img_rgb = np.array(page)
        try:
            aligned_img = align_page(img_rgb, debug=debug)
            aligned_pages.append(aligned_img)
        except ValueError as err:
            print(f"Página {idx + 1}: Error -> {err}")
            aligned_pages.append(None)

    return aligned_pages

def align_page( img_rgb: np.ndarray, margin_w: float = 0.15, margin_h: float = 0.04,
                aspect_tolerance: float = 0.1, debug: bool = False,
                ) -> np.ndarray:
    """
    Detecta las marcas fiduciarias de las 4 esquinas, corrige la perspectiva y binariza.

    Args:
        img_rgb (np.ndarray): Imagen RGB de la página escaneada del formulario E-14.
        margin_w (float, opcional): Proporción del ancho para buscar marcas en las esquinas.
            Por defecto es 0.15.
        margin_h (float, opcional): Proporción del alto para buscar marcas en las esquinas.
            Por defecto es 0.04.
        aspect_tolerance (float, opcional): Tolerancia de relación de aspecto
            para validar que el contorno sea un cuadrado. Por defecto es 0.1.
        debug (bool, opcional): Si es True, renderiza en pantalla las esquinas y la página
            alineada. Por defecto es False.

    Raises:
        ValueError: Si no se logra detectar la marca fiduciaria en alguna de las 4 esquinas.

    Returns:
        np.ndarray: Página alineada geométricamente y binarizada (blanco y negro).
    """
    # Conversión para procesamiento OpenCV
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray_img.shape
    #Margenes en pixeles
    margin_x, margin_y = int(width * margin_w), int(height * margin_h)

    # Zonas de búsqueda en orden estricto: TL, TR, BL, BR
    # Diccionario de zonas de interes recortados ROI y offset
    # "key": imagen[y_inicio:y_fin, x_inicio:x_fin], (offset_x, offset_y) 
    
    corner_zones: dict[str, tuple[np.ndarray, tuple[int, int]]] = {
        "Top-Left": (gray_img[0:margin_y, 0:margin_x], (0, 0)),
        "Top-Right": (gray_img[0:margin_y, width - margin_x : width], (width - margin_x, 0)),
        "Bottom-Left": (gray_img[height - margin_y : height, 0:margin_x], (0, height - margin_y)),
        "Bottom-Right": ( gray_img[height - margin_y : height, width - margin_x : width],
                            (width - margin_x, height - margin_y),
                        ),
    }
    #--------------------------- REVISION --------------------------------------
    # Crear una figura con 2 filas y 2 columnas
    # Mapea las posiciones en la cuadrícula
    if debug:
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        axes_map = {
            "Top-Left": axes[0, 0],
            "Top-Right": axes[0, 1],
            "Bottom-Left": axes[1, 0],
            "Bottom-Right": axes[1, 1],
        }
        for zone_name, (roi, _) in corner_zones.items():
            ax = axes_map[zone_name]
            ax.imshow(roi, cmap="gray")
            ax.set_title(f"{zone_name} ({roi.shape[0]}x{roi.shape[1]} px)")
            ax.axis("off")
        plt.tight_layout()
        plt.show()
    #--------------------------- REVISION --------------------------------------

    source_points = []
    
    for zone_name, (roi, offset) in corner_zones.items():
        #cv2.THRESH_BINARY_INV: Invierte los colores, OpenCV exige que los objetos a detectar sean blancos
        #cv2.THRESH_OTSU: Calcula automáticamente el umbral de separación óptimo entre fondo y figura
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #La función devuelve una tupla (valor_de_umbral, imagen_binarizada).

        #cv2.RETR_EXTERNAL: Extrae los contornos exteriores más externos
        #cv2.CHAIN_APPROX_SIMPLE: Comprime lineas rectas a dos vertices 
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #La función devuelve una lista con los contornos de cada cuadro que encontro

        mark_found = False
        roi_area = roi.shape[0] * roi.shape[1]
        
        for cnt in contours:
            #Calcula el area de los contornos y aplica una restriccion para descartar falsos positivos
            area = cv2.contourArea(cnt)
            if not (0.002 * roi_area < area < 0.3 * roi_area): 
                continue

            #Calcula el perimetro y simplifica la forma a vertices
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

            if len(approx) == 4:
                #Calcula la forma delimitadora para luego probar si es un cuadro 
                #(teniendo en cuenta la tolerancia)
                _, _, w_box, h_box = cv2.boundingRect(approx)
                aspect_ratio = float(w_box) / h_box

                if (1.0 - aspect_tolerance) <= aspect_ratio <= (1.0 + aspect_tolerance):
                    #Se obtiene el centro del cuadrado
                    moments = cv2.moments(cnt)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"]) + offset[0]
                        cy = int(moments["m01"] / moments["m00"]) + offset[1]
                        source_points.append([cx, cy])
                        mark_found = True
                        break

        if not mark_found:
            raise ValueError(f"Fallo geométrico: Marca no detectada en {zone_name}.")

    pts_src = np.array(source_points, dtype=np.float32)

    # Mapeo a rectangulo perfecto (e14) respetando dimensiones originales
    pts_dst = np.array(
        [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]],
        dtype=np.float32,
    )

    #Calcula la matriz de conversion para ajustar la imagen a un rectangulo
    perspective_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)

    #Aplica la matriz para convertir la imagen a un rectangulo
    aligned_img = cv2.warpPerspective(img_rgb, perspective_matrix, (width, height))

    #Prueba de visualizacion de Img
    if debug: 
        plot_aligned_grid(aligned_img) 

    #Se toma la imagen alineada y recortada en los cuadros para aplicar filtro B/N
    return apply_adaptive_threshold(aligned_img, block_size=81, c_constant=40)




def apply_adaptive_threshold( crop_img: np.ndarray, block_size: int = 81, c_constant: int = 40, 
                            ) -> np.ndarray:
    """
    Aplica binarización adaptativa a una imagen para separar trazos del fondo.

    Args:
        crop_img (np.ndarray): Imagen de entrada en formato RGB o escala de grises.
        block_size (int, opcional): Tamaño del vecindario de píxeles para calcular el umbral
            (debe ser impar). Por defecto es 81.
        c_constant (int, opcional): Constante sustraída de la media ponderada.
            Por defecto es 40.

    Returns:
        np.ndarray: Imagen binarizada en 2D (fondo blanco = 255, trazos negros = 0).
    """

    # Validación de matriz bidimensional (Escala de Grises)

    if crop_img.ndim == 3:
        gray_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = crop_img

    # Aplicación del umbral adaptativo. Se hicieron pruebas y se observo que un 
    # punto optimo para los e14 era block_size=81, C=40
    binary_img = cv2.adaptiveThreshold( gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, block_size, c_constant,
                                        )
    
    return binary_img