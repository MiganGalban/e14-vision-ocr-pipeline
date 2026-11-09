import cv2
import numpy as np
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick



def plot_aligned_grid(aligned_img: np.ndarray, width_in: int = 12, 
                        height_in: int = 18) -> None:
    """
    Muestra la imagen alineada con una cuadrícula calibrada en porcentajes (0% a 100%).

    Args:
        aligned_img (np.ndarray): Imagen preprocesada a visualizar.
        width_in (int, opcional): Ancho de la figura en pulgadas. Por defecto es 12.
        height_in (int, opcional): Alto de la figura en pulgadas. Por defecto es 18.
    """
    height, width = aligned_img.shape[:2]
    print(f"Dimensiones de la imagen: Alto = {height}, Ancho = {width}")
    plt.figure(figsize=(width_in, height_in))
    plt.imshow(aligned_img, cmap="gray")
    plt.grid(True, color="red", linestyle="-", linewidth=0.2)


    # Definimos los ticks y agregamos la rotación
    plt.xticks(np.linspace(0, width, 30))
    plt.yticks(np.linspace(0, height, 120))

    plt.gca().xaxis.set_major_formatter( mtick.FuncFormatter(lambda x, _: 
                                        f"{(x / width * 100):.2f}") )

    plt.gca().yaxis.set_major_formatter( mtick.FuncFormatter(lambda y, _:
                                        f"{(y / height * 100):.2f}") )
    
    
    # Configuración para reflejar ejes en los 4 lados
    plt.gca().tick_params( axis="x", labelrotation=90, 
                            top=True, bottom=True, 
                            labeltop=True, labelbottom=True )
    plt.gca().tick_params( axis="y", 
                            left=True, right=True, 
                            labelleft=True, labelright=True )
    
    plt.xlabel("Eje X (Porcentaje)")
    plt.ylabel("Eje Y (Porcentaje)")
    plt.title("Mapa de Píxeles del E-14 Alineado")
    plt.show()

def plot_crop_lite(sample_img: np.ndarray, width_in: int = 6, 
                    height_in: int = 3) -> None:
    """
    Muestra un recorte de imagen individual con cuadrícula en píxeles.

    Args:
        sample_img (np.ndarray): Recorte de imagen a visualizar.
        width_in (int, opcional): Ancho de la figura en pulgadas. Por defecto es 6.
        height_in (int, opcional): Alto de la figura en pulgadas. Por defecto es 3.
    """
    plt.figure(figsize=(width_in, height_in), layout="constrained")
    plt.imshow(sample_img, cmap="gray")
    plt.grid(True, color="red", linestyle="-", linewidth=0.5)

    # Definimos los ticks y agregamos la rotación
    plt.xticks(np.arange(0, sample_img.shape[1], 50), rotation=90)
    plt.yticks(np.arange(0, sample_img.shape[0], 50))

    plt.xlabel("Eje X (Columnas)")
    plt.ylabel("Eje Y (Filas)")
    plt.show()

def plot_three_cells( img_1: np.ndarray, img_2: np.ndarray, img_3: np.ndarray, 
                        title: str = "", width_in: int = 10, height_in: int = 4,
                    ) -> None:
    """
    Muestra en una fila las 3 casillas de dígitos asociadas a un campo de votación.

    Args:
        img_1 (np.ndarray): Recorte de la centena (casilla 1).
        img_2 (np.ndarray): Recorte de la decena (casilla 2).
        img_3 (np.ndarray): Recorte de la unidad (casilla 3).
        title (str, opcional): Título superior del grupo de celdas. Por defecto es "".
        width_in (int, opcional): Ancho de la figura en pulgadas. Por defecto es 10.
        height_in (int, opcional): Alto de la figura en pulgadas. Por defecto es 4.
    """

    images = [img_1, img_2, img_3]
    fig, axes = plt.subplots(1, 3, figsize=(width_in, height_in), layout="constrained")


    for idx, ax in enumerate(axes):
        img = images[idx]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Celda {idx + 1}", pad=30, fontsize=12)

        #Definir posiciones de ticks independientes para cada imagen
        ax.set_xticks(np.linspace(0, img.shape[1], 4))  # 10 ticks en X
        ax.set_yticks(np.linspace(0, img.shape[0], 4))  # 10 ticks en Y

        #Formateadores de porcentaje (im=img asegura el enlace correcto del tamaño)
        ax.xaxis.set_major_formatter( mtick.FuncFormatter(lambda x, _, im=img:
                                        f"{(x / im.shape[1]):.1f}") )
        ax.yaxis.set_major_formatter( mtick.FuncFormatter(lambda y, _, im=img: 
                                        f"{(y / im.shape[0]):.1f}") )

        #Configurar marcas y rotación en eje X (Arriba y Abajo)
        ax.tick_params( axis="x", 
                        top=False, bottom=True, 
                        labeltop=False, labelbottom=True, labelrotation=90 )

        #Configurar marcas en eje Y (Izquierda y Derecha)
        ax.tick_params( axis="y", 
                        left=True, right=False, 
                        labelleft=True, labelright=False  )
        ax.grid(True, color="red", linestyle="-", linewidth=0.5)

    # Ajusta el espacio entre subplots para evitar solapamiento de etiquetas
    #plt.tight_layout(pad=3.0)
    fig.suptitle(title, fontweight="bold")
    plt.show()


def plot_roi_dictionary(crops_dict: dict[str, np.ndarray]) -> None:
    """
    Itera y grafica un diccionario de recortes agrupando casillas triples e individuales.

    Args:
        crops_dict (dict[str, np.ndarray]): Diccionario de zonas generado por `tools.py`.
    """
    common_crops = { "Metadatos_Encabezado", "Nivelacion_Mesa", "QR", "Barras",
                    "Divipole", "Codigo", "Observ_jud", "Re_count_si",
                    "Re_count_no", "Count_req_by", "On_behalf_of",  }
    
    triple_cell_prefixes = { "Votos_A", "Votos_B", "Total_E11", "Total_Urna", "Total_Incinerados", 
                            "V_Blanco", "V_Nulos", "V_Sin_Marcar", "Suma_Total",  }
    rendered_prefixes: set[str] = set()
    for key, crop in crops_dict.items():
        prefix = key[:-2]
        #print(k, ' == ', crops_com)
        if key in common_crops or key.startswith("Firma"):
            plot_crop_lite(crop)
        elif prefix in triple_cell_prefixes and prefix not in rendered_prefixes:
            rendered_prefixes.add(prefix)
            plot_three_cells(   crops_dict[f"{prefix}_1"],
                                crops_dict[f"{prefix}_2"],
                                crops_dict[f"{prefix}_3"],
                                title=prefix,    )
        else:
            print('Problema visualizando diccionario de recortes')

