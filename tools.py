import cv2
import numpy as np
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import re


def com_roi(img_alineada):
    """
    ROI de las zonas comunes entre paginas

    Args:
        img_alineada (numpy_array): La imagen ya preprocesada

    Returns:
        recortes_resultado (Dict): Un diccionario con el nombre de la zona y el recorte
    """
    alto, ancho = img_alineada.shape[:2]
    #print(f"Dimensiones de la imagen: Alto = {alto}, Ancho = {ancho}")
    #Se definen los ROI en porcentajes, para que sean proporciones.
    mapa_coordenadas = {
        "Metadatos_Encabezado":
            (int(alto * 0.026), int(alto * 0.12), int(ancho * 0.242), ancho),
        "QR":
            (int(alto * 0.04), int(alto * 0.12), 0, int(ancho * 0.242)),
        "Barras":
            (int(alto * 0.024), int(alto * 0.038),  int(ancho * 0.35),  int(ancho * 0.65)),
        "Divipole":
            (int(alto * 0.123), int(alto * 0.219), 0, ancho ), 
    }
    recortes_resultado = {}
    for zona, (ymin, ymax, xmin, xmax) in mapa_coordenadas.items():
        recorte = img_alineada[ymin:ymax, xmin:xmax]
        recortes_resultado[zona] = recorte
    return recortes_resultado
    

def roi_p1(img_alineada):
    """
    ROI de las zonas de la pag 1. La mayoria son casillas una a una.
    
        Args:
            img_alineada (numpy_array): La imagen ya preprocesada
    
        Returns:
            recortes_resultado (Dict): Un diccionario con el nombre de la zona y el recorte
    """
    alto, ancho = img_alineada.shape[:2]
    #print(f"Dimensiones de la imagen: Alto = {alto}, Ancho = {ancho}")
    cv1_1= int(ancho * 0.725) #Celda vertical 1 - Primer limite
    cv1_2= int(ancho * 0.825)
    cv2_1= int(ancho * 0.81) #Celda vertical 2 - Primer limite
    cv2_2= int(ancho * 0.91)
    cv3_1= int(ancho * 0.89) #Celda vertical 3 - Primer limite
    cv3_2= int(ancho * 0.99)
    mapa_coordenadas = {

        "Nivelacion_Mesa":
            (int(alto * 0.24), int(alto * 0.34), 0, ancho ),

        #"Total_E11_1": (int(alto * 0.24), int(alto * 0.27), int(ancho * 0.72), ancho ),
        "Total_E11_1": (int(alto * 0.24), int(alto * 0.27), cv1_1, cv1_2 ),
        "Total_E11_2": (int(alto * 0.24), int(alto * 0.27), cv2_1, cv2_2 ),
        "Total_E11_3": (int(alto * 0.24), int(alto * 0.27), cv3_1, cv3_2 ),

        #"Total_Urna": (int(alto * 0.27), int(alto * 0.305), int(ancho * 0.72), ancho ),
        "Total_Urna_1": (int(alto * 0.27), int(alto * 0.30), cv1_1, cv1_2 ),
        "Total_Urna_2": (int(alto * 0.27), int(alto * 0.30), cv2_1, cv2_2 ),
        "Total_Urna_3": (int(alto * 0.27), int(alto * 0.30), cv3_1, cv3_2 ),

        #"Total_Incinerados": (int(alto * 0.305), int(alto * 0.336), int(ancho * 0.72), ancho ),
        "Total_Incinerados_1": (int(alto * 0.30), int(alto * 0.33), cv1_1, cv1_2 ),
        "Total_Incinerados_2": (int(alto * 0.30), int(alto * 0.33), cv2_1, cv2_2 ),
        "Total_Incinerados_3": (int(alto * 0.30), int(alto * 0.33), cv3_1, cv3_2 ),

        #"Votos_1_1": (int(alto * 0.445), int(alto * 0.475), cv1_1, cv1_2 ),
        "Votos_A_1": (int(alto * 0.445), int(alto * 0.475), cv1_1, cv1_2 ),
        "Votos_A_2": (int(alto * 0.445), int(alto * 0.475), cv2_1, cv2_2 ),
        "Votos_A_3": (int(alto * 0.445), int(alto * 0.475), cv3_1, cv3_2 ),

        "Votos_B_1": (int(alto * 0.605), int(alto * 0.635), cv1_1, cv1_2 ),
        "Votos_B_2": (int(alto * 0.605), int(alto * 0.635), cv2_1, cv2_2 ),
        "Votos_B_3": (int(alto * 0.605), int(alto * 0.635), cv3_1, cv3_2 ),

        "V_Blanco_1": (int(alto * 0.723), int(alto * 0.753), cv1_1, cv1_2 ),
        "V_Blanco_2": (int(alto * 0.723), int(alto * 0.753), cv2_1, cv2_2 ),
        "V_Blanco_3": (int(alto * 0.723), int(alto * 0.753), cv3_1, cv3_2 ),

        #"V_Nulos_1": (int(alto * 0.753), int(alto * 0.785), cv1_1, cv1_2 ),
        "V_Nulos_1": (int(alto * 0.753), int(alto * 0.783), cv1_1, cv1_2 ),
        "V_Nulos_2": (int(alto * 0.753), int(alto * 0.783), cv2_1, cv2_2 ),
        "V_Nulos_3": (int(alto * 0.753), int(alto * 0.783), cv3_1, cv3_2 ),

        #"V_Sin_Marcar_1": (int(alto * 0.785), int(alto * 0.817), cv1_1, cv1_2 ),
        "V_Sin_Marcar_1": (int(alto * 0.785), int(alto * 0.815), cv1_1, cv1_2 ),
        "V_Sin_Marcar_2": (int(alto * 0.785), int(alto * 0.815), cv2_1, cv2_2 ),
        "V_Sin_Marcar_3": (int(alto * 0.785), int(alto * 0.815), cv3_1, cv3_2 ),

        "Suma_Total_1": (int(alto * 0.817), int(alto * 0.847), cv1_1, cv1_2 ),
        "Suma_Total_2": (int(alto * 0.817), int(alto * 0.847), cv2_1, cv2_2 ),
        "Suma_Total_3": (int(alto * 0.817), int(alto * 0.847), cv3_1, cv3_2 ),
    }
    recortes_resultado = {}
    for zona, (ymin, ymax, xmin, xmax) in mapa_coordenadas.items():
        recorte = img_alineada[ymin:ymax, xmin:xmax]
        recortes_resultado[zona] = recorte
    return recortes_resultado

def roi_t_p1(img_alineada):
    """Prueba:
    ROI de las zonas de la pag 1, las 3 casillas de interes.
        
            Args:
                img_alineada (numpy_array): La imagen ya preprocesada
        
            Returns:
                recortes_resultado (Dict): Un diccionario con el nombre de la zona y el recorte
    """
    alto, ancho = img_alineada.shape[:2]
    #print(f"Dimensiones de la imagen: Alto = {alto}, Ancho = {ancho}")
    cv1_1= int(ancho * 0.72) #Celda vertical 1 - Primer limite
    cv3_1= int(ancho * 0.89) #Celda vertical 3 - Primer limite
    cv3_2= int(ancho * 0.99)
    mapa_coordenadas = {

        "Nivelacion_Mesa": (int(alto * 0.24), int(alto * 0.34), 0, cv3_2 ),

        "Total_E11": (int(alto * 0.24), int(alto * 0.27), cv1_1, cv3_2 ),
        "Total_Urna": (int(alto * 0.27), int(alto * 0.30), cv1_1, cv3_2 ),
        "Total_Incinerados": (int(alto * 0.30), int(alto * 0.33), cv1_1, cv3_2 ),

        "Votos_A": (int(alto * 0.445), int(alto * 0.475), cv1_1, cv3_2 ),
        "Votos_B": (int(alto * 0.605), int(alto * 0.635), cv1_1, cv3_2 ),

        "V_Blanco": (int(alto * 0.723), int(alto * 0.753), cv1_1, cv3_2 ),
        "V_Nulos": (int(alto * 0.753), int(alto * 0.783), cv1_1, cv3_2 ),
        "V_Sin_Marcar": (int(alto * 0.785), int(alto * 0.815), cv1_1, cv3_2 ),

        "Suma_Total": (int(alto * 0.817), int(alto * 0.847), cv1_1, cv3_2 ),
    }
    recortes_resultado = {}
    for zona, (ymin, ymax, xmin, xmax) in mapa_coordenadas.items():
        recorte = img_alineada[ymin:ymax, xmin:xmax]
        recortes_resultado[zona] = recorte
    return recortes_resultado

def roi_p2(img_alineada):
    """
    ROI de las zonas de la pag 2. Solo sera para saber si el espacio fue diligenciado.

            Args:
                img_alineada (numpy_array): La imagen ya preprocesada
        
            Returns:
                recortes_resultado (Dict): Un diccionario con el nombre de la zona y el recorte
    """
    alto, ancho = img_alineada.shape[:2]
    mapa_coordenadas = {
        "Observ_jud": (int(alto * 0.22), int(alto * 0.59), 0, ancho ),
        "Re_count_si": (int(alto * 0.596), int(alto * 0.614), int(0.415*ancho), int(0.482*ancho) ), ##> 13.5%
        "Re_count_no": (int(alto * 0.596), int(alto * 0.614), int(0.502*ancho), int(0.569*ancho) ), ##> 18.5%
        "Count_req_by": (int(alto * 0.621), int(alto * 0.689), 0, ancho ),
        "On_behalf_of": (int(alto * 0.689), int(alto * 0.756), 0, ancho ),
        "Firma_J1": (int(alto * 0.78), int(alto * 0.823), 0, int(0.48*ancho) ),
        "Firma_J3": (int(alto * 0.851), int(alto * 0.894), 0, int(0.48*ancho) ),
        "Firma_J5": (int(alto * 0.923), int(alto * 0.966), 0, int(0.48*ancho) ),
        "Firma_J2": (int(alto * 0.78), int(alto * 0.823), int(0.50*ancho), int(0.98*ancho) ),
        "Firma_J4": (int(alto * 0.851), int(alto * 0.894), int(0.50*ancho), int(0.98*ancho) ),
        "Firma_J6": (int(alto * 0.923), int(alto * 0.966), int(0.50*ancho), int(0.98*ancho) ),

        "Firma_cc_J1": (int(alto * 0.823), int(alto * 0.838), int(0.065*ancho), int(0.48*ancho) ),
        "Firma_cc_J3": (int(alto * 0.894), int(alto * 0.909), int(0.065*ancho), int(0.48*ancho) ),
        "Firma_cc_J5": (int(alto * 0.966), int(alto * 0.981), int(0.065*ancho), int(0.48*ancho) ),
        "Firma_cc_J2": (int(alto * 0.823), int(alto * 0.838), int(0.565*ancho), int(0.98*ancho) ),
        "Firma_cc_J4": (int(alto * 0.894), int(alto * 0.909), int(0.565*ancho), int(0.98*ancho) ),
        "Firma_cc_J6": (int(alto * 0.966), int(alto * 0.981), int(0.565*ancho), int(0.98*ancho) ),
    }

    recortes_resultado = {}
    for zona, (ymin, ymax, xmin, xmax) in mapa_coordenadas.items():
        recorte = img_alineada[ymin:ymax, xmin:xmax]
        recortes_resultado[zona] = recorte
    return recortes_resultado

# ---------------------------------------------------------------------------# ---------------------------------------------------------------------------

def extraer_divipole(texto_ocr):
    """
    Extrae variables estructuradas de una cadena plana OCR.
    Las expresiones regulares para campos numéricos incluyen [0-9], 'O' y 'o' 
    para prever y capturar los errores de lectura del OCR antes de la coerción.

    Args:
        texto_ocr (str): Texto para extraer variables

    Returns:
        variables (Dict): Diccionario con las variables de Divipole y su valor.
    """
    
    
    patrones = {
        "DP_Dept": r"DEPARTAMENTO:\s*([\doO]+)",
        "DP_Munp": r"MUNICIPIO:\s*([\doO]+)",
        "DP_Zona": r"ZONA:\s*([\doO]+)",
        "DP_Psto": r"PUESTO:\s*([\doO]+)",
        "DP_Mesa": r"MESA:\s*([\doO]+)",
        "DP_Lugr": r"LUGAR:\s*(.+?)(?=$)" # Captura desde "LUGAR:" hasta el final
    }
    
    variables = {}
    
    for clave, patron in patrones.items():
        match = re.search(patron, texto_ocr, re.IGNORECASE)
        
        if match:
            valor_capturado = match.group(1).strip()
            
            # Coerción: Aplicar reemplazo O->0 exclusivamente a campos numéricos
            if clave != "DP_Lugr":
                valor_saneado = valor_capturado.upper().replace('O', '0')
                variables[clave] = valor_saneado
            else:
                variables[clave] = valor_capturado
        else:
            variables[clave] = None
            
    return variables

