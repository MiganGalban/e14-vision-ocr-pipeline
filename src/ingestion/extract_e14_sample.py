
from dotenv import load_dotenv
from supabase import create_client
import pandas as pd
import math

import os
from pathlib import Path

def fetch_all_rows(supabase, table: str, columns: str, page_size: int = 1000) -> list[dict]:
    """Supabase pagina resultados (limite ~1000 por request por defecto).
    Esta función pagina hasta traer todo el universo de puestos."""
    all_rows = []
    start = 0
    while True:
        # Se genera la Query para supabase
        resp = (
            supabase.table(table).select(columns)
            .range(start, start + page_size - 1)
            .execute()
        )
        # Se guardan los datos adquiridos en la variable rows, si esta vacio rompe el ciclo
        rows = resp.data
        if not rows:
            break
        # Se acumulan los datos obtenidos en la consulta y se comprueba si ya eran los ultimos datos
        all_rows.extend(rows)
        if len(rows) < page_size:
            break

        start += page_size

    return all_rows



def get_stratum(zz: str | int) -> str:
    """Clasifica una zona geográfica en una categoría o estrato operativo.

    Aplica la regla de codificación de zonas (zz) de la Registraduría:
    - zz = 99: Zona Rural (veredas, corregimientos).
    - zz = 98: Cárcel (puestos especiales penitenciarios).
    - Cualquier otro valor: Zona Urbana (cabeceras municipales o comunas).

    Args:
        zz (str | int): Código numérico o alfanumérico que representa la zona.
            Debe ser convertible a entero mediante `int()`.

    Returns:
        str: Etiqueta del estrato correspondiente ('rural', 'carcel' o 'urbano').
    """
    zz = int(zz)
    if zz == 99:
        return "rural"
    elif zz == 98:
        return "carcel"
    else:
        return "urbano"

# Se calcula el tamaño de muestra (n) con la fórmula de Cochran con corrección por población finita
# Se asume Probabilidad p y q de 50% por que no se conoce su valor

def get_sample_size(N: int, confidence: float = 0.95, margin: float = 0.05, p: float = 0.5) -> int:
    """Calcula el tamaño de muestra (n) mediante la fórmula de Cochran con corrección por población finita.

    Calcula n bajo el supuesto de máxima varianza (p = q = 0.5 por defecto)
    utilizando un factor de corrección para poblaciones finitas:
        n = (N * z^2 * p * q) / (margin^2 * (N - 1) + z^2 * p * q)

    Args:
        N (int): Tamaño total del universo o población finita.
        confidence (float, opcional): Nivel de confianza estadístico. Valores
            soportados en la tabla z interna: 0.90 (z=1.645), 0.95 (z=1.96), 0.99 (z=2.576).
            Por defecto es 0.95.
        margin (float, opcional): Margen de error máximo permitido en valor decimal.
            Por defecto es 0.05 (±5%).
        p (float, opcional): Proporción esperada del fenómeno de interés.
            Por defecto es 0.5.

    Returns:
        int: Número mínimo de observaciones a muestrear, redondeado al entero
        superior más cercano mediante `math.ceil()`.
    """
    z_table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_table.get(confidence, 1.96)
    q=p
    n = (N*(z**2)*p*q)/((margin**2)*(N-1)+(z**2)*p*q)

    # Otra forma partiendo desde el supuesto de infinito
    # n0 = (z ** 2 * p * (1 - p)) / (margin ** 2)
    # n = n0 / (1 + (n0 - 1) / N)

    return math.ceil(n)

# Se contruye el marco muestral a nivel de MESA (no de puesto)
# Cada mesa física se convierte en una fila en el dataframe
def sample_voting_tables(df: pd.DataFrame, stratum_allocation: pd.Series) -> pd.DataFrame:
    """Desagrega puestos a nivel de mesa y ejecuta un muestreo aleatorio estratificado.

    La función realiza dos operaciones secuenciales:
    1. Expande el marco muestral desde el nivel de puesto hacia el nivel atómico
        de mesa individual, creando una fila por cada mesa reportada en `mesas_domingo`.
    2. Ejecuta un Muestreo Aleatorio Simple (MAS) sin reemplazo dentro de cada estrato,
        extrayendo la cantidad exacta de mesas indicada en la serie de asignación.

    Args:
        df (pd.DataFrame): DataFrame con información de puestos de votación. Debe
            contener las columnas: 'dd', 'mm', 'zz', 'pp', 'departamento',
            'municipio', 'puesto', 'estrato' y 'mesas_domingo'.
        stratum_allocation (pd.Series): Serie de pandas cuyo índice corresponde al nombre
            del estrato (str) y sus valores corresponden al tamaño de muestra
            entero (int) requerido para dicho estrato.

    Returns:
        pd.DataFrame: Conjunto de datos final con las mesas seleccionadas aleatoriamente,
        con índice reiniciado y conservando las variables de ubicación y estrato.
    """
    frames = []
    for _, r in df.iterrows(): #No hay necesidad de guardar el indice
        # Si el valor no es nulo guarde el valor de lo contrario guarde un 0 en num_voting_tables
        num_voting_tables = int(r["mesas_domingo"]) if pd.notna(r["mesas_domingo"]) else 0
        # Se guarda una lista de diccionarios con los datos de la fila, y se van
        # generando filas segun "num_voting_tables"

        for table_idx in range(1, num_voting_tables + 1):
            frames.append({
                "dd": r["dd"], "mm": r["mm"], "zz": r["zz"], "pp": r["pp"],
                "departamento": r["departamento"], "municipio": r["municipio"],
                "puesto": r["puesto"], "mesa": table_idx, "estrato": r["estrato"],
            })

    # Se pasa la lista de diccionarios a DataFrame, donde cada diccionario es una fila
    voting_tables_df = pd.DataFrame(frames)
    #  Se genera el muestreo aleatorio simple dentro de cada estrato
    SEED = 42 #Dejamos una semilla por si se quiere replicar el experimento
    sections = []

    # Como "stratum_allocation" es una serie, "estrato" es el idx y "value_stratum" el valor
    # (voting_tables_df["estrato"] == estrato) Genera una mascara booleana para mostrar
    # solo las filas de ese estrato y luego tomar muestras (value_stratum) "aleatoriamente"
    for stratum, value_stratum in stratum_allocation.items():
        group = voting_tables_df[ voting_tables_df["estrato"] == stratum ]
        sections.append(group.sample(n=value_stratum, random_state=SEED))

    sample_df = pd.concat(sections).reset_index(drop=True)

    return sample_df


def generate_e14_sample( table_supabase: str = "divipole_regis", 
            columns: str = "dd,mm,zz,pp,departamento,municipio,puesto,comuna,mesas_domingo",
            confidence: float = 0.95, margin_error: float = 0.05, output_path: Path | str = None,
            ) -> pd.DataFrame:
    """Orquesta el pipeline completo de extracción, estratificación, muestreo y exportación.

    Ejecuta el flujo integral:
    1. Autenticación y conexión con el cliente Supabase vía variables de entorno.
    2. Extracción paginada de DIVIPOLE y exclusión del universo consular (`dd == '88'`).
    3. Clasificación de zonas (`urbano`, `rural`, `carcel`) y cálculo del tamaño
        de muestra global mediante la fórmula de Cochran con corrección por población finita.
    4. Asignación proporcional de cuotas por estrato con corrección de residuos por redondeo.
    5. Expansión a nivel mesa, muestreo aleatorio simple estratificado y persistencia en CSV.

    Args:
        table_supabase (str, opcional): Nombre de la tabla DIVIPOLE en Supabase.
            Por defecto es "divipole_regis".
        columns (str, opcional): Cadena separada por comas con los campos requeridos.
            Por defecto incluye códigos DIVIPOLE, geografía y conteo de mesas.
        confidence (float, opcional): Nivel de confianza estadística (0.90, 0.95, 0.99).
            Por defecto es 0.95.
        margin_error (float, opcional): Margen de error muestral tolerable en decimal.
            Por defecto es 0.05 (±5%).
        output_path (Path | str | None, opcional): Ruta de destino para guardar el archivo
            CSV generado. Si es None, se asigna por defecto a la ruta relativa
            `../../data/muestra_e14_segunda_vuelta.csv`.

    Raises:
        ValueError: Si las variables de entorno `SUPABASE_URL` o `SUPABASE_KEY` no están
            definidas en el archivo `.env` o en el entorno del sistema.

    Returns:
        pd.DataFrame: DataFrame con la muestra aleatoria final de mesas seleccionadas.
    """
    # Autenticación y conexión con el cliente Supabase 
    load_dotenv()
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError("Variables SUPABASE_URL o SUPABASE_KEY no definidas en el entorno.")

    supabase = create_client(supabase_url, supabase_key)

    # Extracción y filtrado
    rows = fetch_all_rows(supabase, table_supabase, columns)
    df = pd.DataFrame(rows)
    df = df[df["dd"] != "88"].copy()

    # Estratificación y cálculo de muestra (sample_df)
    df["estrato"] = df["zz"].apply(get_stratum)
    N = df["mesas_domingo"].sum()
    n = get_sample_size(N, confidence=confidence, margin=margin_error)

    # Asignación proporcional con ajuste de redondeo
    stratum_counts = df.groupby("estrato")["mesas_domingo"].sum()
    stratum_allocation = (stratum_counts / N * n).round().astype(int)
    rounding_diff = n - stratum_allocation.sum()
    if rounding_diff != 0:
        stratum_allocation[stratum_counts.idxmax()] += rounding_diff

    # Muestreo y exportación
    sample_df = sample_voting_tables(df, stratum_allocation)

    # Sube dos niveles desde src/ingestion/ hasta la raíz y concatena "data"
    #   __file__               -->  .../e14/src/ingestion/e14_extraccion_muestra.py
    #   .resolve().parents[0]  -->  .../e14/src/ingestion
    #   .resolve().parents[1]  -->  .../e14/src
    #   .resolve().parents[2]  -->  .../e14 (raíz)
    #   / "data"               -->  .../e14/data
    #   / "muestra_e14_segunda_vuelta.csv"

    if output_path is None:
        output_path = ( Path(__file__).resolve().parents[2]
                    / "data"
                    / "muestra_e14_segunda_vuelta.csv"
                    )

    sample_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return sample_df

