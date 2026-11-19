from pathlib import Path

import cv2
import pymupdf
import matplotlib.pyplot as plt
import numpy as np

from src.processing.view import plot_aligned_grid


def pdf_to_images(
    pdf_path: str | Path,
    dpi: int = 300,
    debug: bool = False,
) -> list[np.ndarray | None]:
    """Rasteriza un archivo PDF a imágenes RGB de alta resolución y alinea cada página.

    Args:
        pdf_path (str | Path): Ruta del archivo PDF a procesar.
        dpi (int, opcional): Resolución de rasterización en puntos por pulgada. Por defecto es 300.
        debug (bool, opcional): Si es True, renderiza las ventanas de búsqueda de esquinas y la
            página alineada con grilla porcentual. Por defecto es False.

    Returns:
        list[np.ndarray | None]: Lista de matrices 2D alineadas y binarizadas por página
        (o None en la posición correspondiente si falló la detección geométrica).
    """
    # Cálculo del factor de escala respecto a la resolución nativa del estándar PDF (72 DPI)
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    aligned_pages: list[np.ndarray | None] = []

    # Apertura segura del PDF y rasterización página por página directamente en memoria RAM
    with pymupdf.open(str(pdf_path)) as doc:
        for idx, page in enumerate(doc):
            pix = page.get_pixmap(matrix=matrix, colorspace=pymupdf.csRGB, alpha=False)
            img_rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

            try:
                aligned_img = align_page(img_rgb, debug=debug)
                aligned_pages.append(aligned_img)
            except ValueError as err:
                print(f"Página {idx + 1}: Error -> {err}")
                aligned_pages.append(None)

    return aligned_pages


def align_page(
    img_rgb: np.ndarray,
    margin_w: float = 0.15,
    margin_h: float = 0.04,
    aspect_tolerance: float = 0.1,
    debug: bool = False,
) -> np.ndarray:
    """Detecta las 4 marcas fiduciarias de las esquinas, corrige la perspectiva y binariza.

    Args:
        img_rgb (np.ndarray): Imagen RGB de la página escaneada del formulario E-14.
        margin_w (float, opcional): Proporción del ancho para delimitar la ventana de búsqueda
            en cada esquina. Por defecto es 0.15.
        margin_h (float, opcional): Proporción del alto para delimitar la ventana de búsqueda
            en cada esquina. Por defecto es 0.04.
        aspect_tolerance (float, opcional): Desviación permitida respecto a una relación de
            aspecto cuadrada (1.0 ± tolerancia). Por defecto es 0.1.
        debug (bool, opcional): Activa las gráficas de diagnóstico visual. Por defecto es False.

    Raises:
        ValueError: Si no se logra detectar la marca fiduciaria en alguna de las 4 esquinas.

    Returns:
        np.ndarray: Página corregida geométricamente y binarizada en 2D (blanco y negro).
    """
    # Conversión a escala de grises y cálculo de márgenes de búsqueda en píxeles
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray_img.shape
    margin_x, margin_y = int(width * margin_w), int(height * margin_h)

    # Recorte de las 4 esquinas en orden estricto (TL, TR, BL, BR) junto con su offset global
    corner_zones: dict[str, tuple[np.ndarray, tuple[int, int]]] = {
        "Top-Left": (gray_img[0:margin_y, 0:margin_x], (0, 0)),
        "Top-Right": (gray_img[0:margin_y, width - margin_x : width], (width - margin_x, 0)),
        "Bottom-Left": (gray_img[height - margin_y : height, 0:margin_x], (0, height - margin_y)),
        "Bottom-Right": (
            gray_img[height - margin_y : height, width - margin_x : width],
            (width - margin_x, height - margin_y),
        ),
    }

    # Visualización diagnóstica opcional de las 4 zonas de búsqueda
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

    # Detección del centroide de cada marca fiduciaria cuadrada
    source_points: list[list[int]] = []

    for zone_name, (roi, offset) in corner_zones.items():
        # Binarización inversa con umbral automático Otsu (marcas oscuras -> píxeles blancos)
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        mark_found = False
        roi_area = roi.shape[0] * roi.shape[1]

        for cnt in contours:
            # Filtro por área para descartar ruido menor o bloques de borde
            area = cv2.contourArea(cnt)
            if not (0.002 * roi_area < area < 0.3 * roi_area):
                continue

            # Aproximación poligonal para validar geometría de 4 vértices y proporción cuadrada
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

            if len(approx) == 4:
                _, _, w_box, h_box = cv2.boundingRect(approx)
                aspect_ratio = float(w_box) / h_box

                if (1.0 - aspect_tolerance) <= aspect_ratio <= (1.0 + aspect_tolerance):
                    # Cálculo del centroide en coordenadas globales de la página
                    moments = cv2.moments(cnt)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"]) + offset[0]
                        cy = int(moments["m01"] / moments["m00"]) + offset[1]
                        source_points.append([cx, cy])
                        mark_found = True
                        break

        if not mark_found:
            raise ValueError(f"Fallo geométrico: Marca no detectada en {zone_name}.")

    # Transformación homográfica de perspectiva hacia un rectángulo exacto del tamaño original
    pts_src = np.array(source_points, dtype=np.float32)
    pts_dst = np.array(
        [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]],
        dtype=np.float32,
    )
    perspective_matrix = cv2.getPerspectiveTransform(pts_src, pts_dst)
    aligned_img = cv2.warpPerspective(img_rgb, perspective_matrix, (width, height))

    if debug:
        plot_aligned_grid(aligned_img)

    # Binarización adaptativa final optimizada para escaneos con iluminación irregular
    return apply_adaptive_threshold(aligned_img, block_size=81, c_constant=40)


def apply_adaptive_threshold(
    crop_img: np.ndarray,
    block_size: int = 81,
    c_constant: int = 40,
) -> np.ndarray:
    """Aplica binarización adaptativa por media local para separar tinta manuscrita del fondo.

    Args:
        crop_img (np.ndarray): Imagen de entrada en formato RGB o escala de grises.
        block_size (int, opcional): Tamaño impar del vecindario de píxeles para calcular el
            umbral local. Por defecto es 81.
        c_constant (int, opcional): Constante sustraída a la media local para ajustar la
            sensibilidad frente a ruido y sombras. Por defecto es 40.

    Returns:
        np.ndarray: Matriz binarizada en 2D (fondo blanco = 255, trazos negros = 0).
    """
    # Normalización a escala de grises (1 canal) si la imagen ingresa en RGB
    if crop_img.ndim == 3:
        gray_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = crop_img

    # Umbralización adaptativa calibrada para formularios electorales E-14
    binary_img = cv2.adaptiveThreshold(
        gray_img,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_constant,
    )

    return binary_img