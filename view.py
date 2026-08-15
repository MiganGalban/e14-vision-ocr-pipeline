import cv2
import numpy as np
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick



def view_all_perc(img_alineada, w=12, h=18):
    """
    Muestra la imagen con grid en porcentajes por cada lado.

    Args:
        img_alineada (numpy_array): La imagen ya preprocesada
        w (int, optional): Ancho de la figura. Defaults to 12.
        h (int, optional): Alto de la figura. Defaults to 18.
    """
    alto, ancho = img_alineada.shape[:2]
    print(f"Dimensiones de la imagen: Alto = {alto}, Ancho = {ancho}")
    plt.figure(figsize=(w, h))
    plt.imshow(img_alineada, cmap='gray')
    plt.grid(True, color='red', linestyle='-', linewidth=0.2)

    # Definimos los ticks y agregamos la rotación
    plt.xticks(np.linspace(0, img_alineada.shape[1], 30 ))
    plt.yticks(np.linspace(0, img_alineada.shape[0], 120))

    plt.gca().xaxis.set_major_formatter( mtick.FuncFormatter(lambda x, pos:
                            f"{(x / img_alineada.shape[1] * 100):.2f}"))

    plt.gca().yaxis.set_major_formatter( mtick.FuncFormatter(lambda y, pos:
                            f"{(y / img_alineada.shape[0] * 100):.2f}"))
    
    
    # Configuración para reflejar ejes en los 4 lados
    plt.gca().tick_params(axis="x", labelrotation=90,
                        top=True, bottom=True,
                        labeltop=True, labelbottom=True,)
    plt.gca().tick_params(axis="y",
                        left=True, right=True,
                        labelleft=True, labelright=True,)
    
    plt.xlabel("Eje X (Porcentaje)")
    plt.ylabel("Eje Y (Porcentaje)")
    plt.title("Mapa de Píxeles del E-14 Alineado (Etiquetas Rotadas)")
    plt.show()

def view_lite(img_muestra, w=6, h=3):
    """
    Muestra la imagen con grid, en pixeles, predef para algo mas sencillo.

    Args:
        img_alineada (numpy_array): La imagen ya preprocesada
        w (int, optional): Ancho de la figura. Defaults to 6.
        h (int, optional): Alto de la figura. Defaults to 3.    
    """
    plt.figure(figsize=(w, h), layout='constrained')
    plt.imshow(img_muestra, cmap='gray')
    plt.grid(True, color='red', linestyle='-', linewidth=0.5)

    # Definimos los ticks y agregamos la rotación
    plt.xticks(np.arange(0, img_muestra.shape[1], 50), rotation=90)
    plt.yticks(np.arange(0, img_muestra.shape[0], 50))

    plt.xlabel("Eje X (Columnas)")
    plt.ylabel("Eje Y (Filas)")
    plt.show()

def view_cells(img_1, img_2, img_3, titulo="", w=10, h=4 ):
    """
    Muestra las 3 celdas asociadas en una sola imagen.
    Args:
        img_1 (numpy_array): La imagen de la celda ya preprocesada
        img_2 (numpy_array): La imagen de la celda ya preprocesada
        img_3 (numpy_array): La imagen de la celda ya preprocesada
        titulo (str, optional): El titulo de la seccion. Defaults to "".
        w (int, optional): Ancho de la figura. Defaults to 10.
        h (int, optional): Alto de la figura.. Defaults to 4.
    """

    imagenes = [img_1, img_2, img_3]
    fig, axes = plt.subplots(1, 3, figsize=(w, h), layout='constrained')


    for i, ax in enumerate(axes):
        img = imagenes[i]
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Celda {i+1}", pad=30, fontsize=12)

        #Definir posiciones de ticks independientes para cada imagen
        ax.set_xticks(np.linspace(0, img.shape[1], 4))  # 10 ticks en X
        ax.set_yticks(np.linspace(0, img.shape[0], 4))  # 10 ticks en Y

        #Formateadores de porcentaje (im=img asegura el enlace correcto del tamaño)
        ax.xaxis.set_major_formatter( mtick.FuncFormatter(
                            lambda x, pos, im=img: f"{(x / im.shape[1]):.1f}") )
        ax.yaxis.set_major_formatter( mtick.FuncFormatter(
                            lambda y, pos, im=img: f"{(y / im.shape[0]):.1f}") )

        #Configurar marcas y rotación en eje X (Arriba y Abajo)
        ax.tick_params( axis="x", top=False, bottom=True, 
                        labeltop=False, labelbottom=True, labelrotation=90, )

        #Configurar marcas en eje Y (Izquierda y Derecha)
        ax.tick_params( axis="y", left=True, right=False, 
                        labelleft=True, labelright=False, )
        ax.grid(True, color='red', linestyle='-', linewidth=0.5)

    # Ajusta el espacio entre subplots para evitar solapamiento de etiquetas
    #plt.tight_layout(pad=3.0)
    fig.suptitle(titulo, fontweight='bold')
    plt.show()


def view_crops(dict_cut):
    """
    Funcion para gestionar la visualizacion del diccionario que sale de los recortes.

    Args:
        dict_cut (Dict): Un diccionario con el nombre de la zona y el recorte.
    """

    crops_com  = ['Metadatos_Encabezado', 'Nivelacion_Mesa', 'QR', 'Barras', 'Divipole', 'Codigo',
                    'Observ_jud', 'Re_count_si', 'Re_count_no', 'Count_req_by', 'On_behalf_of']
    crops_cell = ['Votos_1', 'Votos_2', 'Total_E11', 'Total_Urna', 'Total_Incinerados', 
                    'V_Blanco', 'V_Nulos', 'V_Sin_Marcar', 'Suma_Total']
    save_cell = set()
    for k, v in dict_cut.items():
        #print(k, ' == ', crops_com)
        if k in crops_com:
            view_lite(dict_cut[k])
        elif (k[:-3] == 'Firma') or (k[:-6] == 'Firma') :
            view_lite(dict_cut[k])
        elif (k[:-2] in crops_cell) and (k[:-2] not in save_cell):
            save_cell.add(k[:-2])
            view_cells(dict_cut[k[:-2]+'_1'], dict_cut[k[:-2]+'_2'], dict_cut[k[:-2]+'_3'], k)
        elif k[:-2] in save_cell:
            print(k, 'Ya se imprimio')
        else:
            print('Problema visualizando diccionario de recortes')

