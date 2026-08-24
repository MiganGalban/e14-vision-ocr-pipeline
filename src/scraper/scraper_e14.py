import os
import csv
import json
import time
import random
import threading
from collections import defaultdict
from curl_cffi import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuración Base ---
import threading
from pathlib import Path

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

# Sube dos niveles desde src/scraper/ hasta la raíz y concatena "data"
#   __file__               -->  .../e14/src/scraper/scraper_e14.py
#   .resolve().parents[0]  -->  .../e14/src/scraper
#   .resolve().parents[1]  -->  .../e14/src
#   .resolve().parents[2]  -->  .../e14 (raíz)
#   / "data"               -->  .../e14/data
BASE_DIR = Path(__file__).resolve().parents[2] / "data"

os.makedirs(BASE_DIR, exist_ok=True)

CACHE_DIR = os.path.join(BASE_DIR, ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# CSV que exportó el notebook de Colab (muestra_e14_segunda_vuelta.csv).
# Ajusta esta ruta al lugar donde lo guardaste localmente.
ERRORS_CSV_PATH = os.path.join(BASE_DIR, "errores_descarga.csv")
SAMPLE_CSV_PATH = os.path.join(BASE_DIR, "muestra_e14_segunda_vuelta.csv")



BASE_URL = "https://escrutinios2vueltapresidente2026.registraduria.gov.co/"
MAX_DOWNLOAD_WORKERS = 2  # Restringido a 1-2 para streaming a Google Drive

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-ES,es;q=0.9",
    "Sec-Ch-Ua": '"Google Chrome";v="125", "Chromium";v="125", "Not.A/Brand";v="24"',
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin"
}

_lock_csv = threading.Lock()


def init_error_log() -> None:
    """Crea el archivo CSV de registro de fallos si no existe e inserta los encabezados.

    Verifica la existencia de "ERRORS_CSV_PATH" en el sistema de archivos local y,
    en caso negativo, escribe la cabecera correspondiente:
    ['Depto', 'Muni', 'Zona', 'Puesto', 'Mesa', 'Error', 'URL'].
    """ 
    if not os.path.exists(ERRORS_CSV_PATH):
        with open(ERRORS_CSV_PATH, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Depto", "Muni", "Zona", "Puesto", "Mesa", "Error", "URL"])


def log_error(meta: tuple, type_error: str, url: str) -> None:
    """Registra una incidencia de descarga o resolución en el archivo CSV de errores.

    Utiliza una exclusión mutua ("_lock_csv") para garantizar escrituras
    atómicas y seguras entre hilos concurrentes (thread-safe).

    Args:
        meta (Tuple[str, ...]): Tupla con identificadores DIVIPOLE:
            (Depto, Muni, Zona, Puesto, Mesa).
        type_error (str): Clasificación o etiqueta del fallo ocurrido.
        url (str): Endpoint donde se presentó la anomalía.
    """
    with _lock_csv:
        with open(ERRORS_CSV_PATH, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(list(meta) + [type_error, url])


def normalize_code(code) -> str:
    """Normaliza un código (dd, mm, zz, pp) quitando ceros a la izquierda,
    para poder comparar el CSV de selección contra las llaves reales del
    JSON de la Registraduría sin depender de que el padding coincida
    (ej. '016' del CSV vs '16' en el JSON, o viceversa).
    Si el valor no es enteramente numérico, remueve espacios y convierte a mayúsculas.

    Args:
        code (Any): Valor a normalizar (entero, string numérico o alfanumérico).

    Returns:
        str: Representación normalizada del código.
    """
    s = str(code).strip().upper()
    if s.isdigit():
        return str(int(s))
    return s


def load_selection(csv_path: str) -> dict:
    """Lee el CSV exportado por Colab (columnas dd,mm,zz,pp,...,mesa,...)
    y arma un diccionario: {(dd,mm,zz,pp) normalizados: {numeros_de_mesa}} 
    con las mesas requeridas.

    Args:
        csv_path (str): Ruta local hacia el archivo CSV de entrada.

    Returns:
        Dict[Tuple[str, str, str, str], Set[int]]: Diccionario donde la llave
            es la tupla `(dd, mm, zz, pp)` normalizada y el valor es el conjunto
            de identificadores numéricos de mesas requeridas para ese puesto.

    Raises:
        FileNotFoundError: Si el archivo especificado en `csv_path` no existe.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No se encontró el CSV de selección en: {csv_path}\n"
            "Copia ahí el archivo que exportó Colab (muestra_e14_segunda_vuelta.csv) "
            "o ajusta la variable SAMPLE_CSV_PATH."
        )

    seletion = defaultdict(set)
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                normalize_code(row["dd"]),
                normalize_code(row["mm"]),
                normalize_code(row["zz"]),
                normalize_code(row["pp"]),
            )
            seletion[key].add(int(row["mesa"]))

    total_tables = sum(len(v) for v in seletion.values())
    print(f"[Selección] {len(seletion)} puesto(s) único(s) | {total_tables} mesa(s) a descargar")
    return seletion


def fetch_cached_json(url: str, local_filename: str):
    """Recupera un archivo JSON desde el almacenamiento local o vía solicitud HTTP.

    Si el archivo existe en `.cache`, se carga directamente de disco. En caso
    contrario (o si está corrupto), se descarga mediante `curl_cffi` impersonando
    un navegador web estándar y se persiste localmente.

    Args:
        url (str): Dirección web de la API o endpoint JSON.
        local_filename (str): Nombre del archivo para persistencia en caché.

    Returns:
        Optional[Union[Dict[str, Any], List[Any]]]: Datos JSON deserializados
            como diccionario o lista, o `None` en caso de error HTTP/red.
    """
    # Obtiene el JSON del almacenamiento local; si no existe, lo descarga y guarda.
    local_path = os.path.join(CACHE_DIR, local_filename)
    
    if os.path.exists(local_path):
        print(f"  [Caché] Leyendo localmente: {local_filename}")
        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"  [!] Archivo corrupto en caché, reintentando descarga: {local_filename}")
            os.remove(local_path)

    print(f"  [Red] Descargando desde API: {url}")
    try:
        resp = requests.get(url, headers=HEADERS, impersonate="chrome110", timeout=30.0)
        if resp.status_code == 200:
            datos = resp.json()
            with open(local_path, 'w', encoding='utf-8') as f:
                json.dump(datos, f)
            return datos
        print(f"  [!] HTTP {resp.status_code} en: {url}")
        return None
    except Exception as e:
        print(f"  [!] Error de red en {url}: {str(e)}")
        return None


def download_pdf(url_pdf: str, destination_path: str, identifier: str, meta: tuple) -> str:
    """Descarga y valida la integridad de un archivo PDF de acta electoral.

    Aplica un retraso aleatorio (0.5 a 1.5 s) para mitigar bloqueos de tasa (rate limiting).
    Valida la firma mágica binaria (`%PDF`) del archivo recibido antes de persistirlo.
    En caso de error, lo anota en el archivo de registro.

    Args:
        url_pdf (str): Enlace directo al documento PDF en el servidor remoto.
        destination_path (str): Ruta del sistema de archivos local para guardar el PDF.
        identifier (str): Etiqueta descriptiva para trazabilidad en logs de consola.
        meta (Tuple[str, ...]): Metadatos de ubicación geográfica de la mesa.

    Returns:
        str: Mensaje descriptivo con el estado final de la operación
            (Existente, Descargado, Falso PDF o Fallo de red).
    """
    if os.path.exists(destination_path):
        return f"⏩ [PDF] Existente: {identifier}"

    time.sleep(random.uniform(0.5, 1.5))

    try:
        resp = requests.get(url_pdf, headers=HEADERS, impersonate="chrome110", timeout=20.0)
        if resp.status_code == 200 and resp.content.startswith(b'%PDF'):
            with open(destination_path, 'wb') as f:
                f.write(resp.content)
            return f"✅ [PDF] Descargado: {identifier}"
        else:
            log_error(meta, f"HTTP_{resp.status_code}_Falso_PDF", url_pdf)
            return f"❌ [PDF] Falso PDF / HTTP {resp.status_code}: {identifier}"
    except Exception as e:
        log_error(meta, "Timeout_Descarga", url_pdf)
        return f"⚠️ [PDF] Fallo de red: {identifier}"


def execute_scraper(SELECTION_CSV: str = SAMPLE_CSV_PATH):
    """Orquesta la ejecución global del proceso de extracción en tres fases.

    - **Fase 1**: Inicializa registros de error y obtiene los archivos maestros
      (`index.json` y estructura `divipole.json`).
    - **Fase 2**: Recorre el árbol jerárquico (depto -> muni -> zona -> puesto)
      filtrando estrictamente los nodos presentes en el CSV de muestra, resuelve
      las rutas relativas de los PDFs correspondientes a cada mesa y construye la
      cola de descargas.
    - **Fase 3**: Ejecuta un pool de hilos (`ThreadPoolExecutor`) para procesar
      las descargas en paralelo y genera el reporte final de completitud.

    Args:
        SELECTION_CSV (str, optional): Ruta al archivo CSV con la muestra de
            mesas. Por defecto usa `SAMPLE_CSV_PATH`.
    """
    init_error_log()
    seletion = load_selection(SELECTION_CSV)

    print("\n[=== FASE 1: DESCARGA DE ESTRUCTURAS MAESTRAS ===]")
    index_map = fetch_cached_json(f"{BASE_URL}data/index.json", "index.json")
    divipole = fetch_cached_json(
        f"{BASE_URL}data/esc/v1/divipole/divipole_20260609_095453_471.json", "divipole.json"
    )

    if not index_map or not divipole:
        print("[-] Fallo crítico: No se obtuvo la estructura base.")
        return

    corporacion = "001"
    pdf_tasks = []
    positions_not_found = []
    visited_keys = set()

    print("\n[=== FASE 2: RESOLUCIÓN DE RUTAS (SOLO PUESTOS SELECCIONADOS) ===]")
    departamentos = divipole.get("departamentos", {})
    for dept_code, dept_data in departamentos.items():
        dept_norm = normalize_code(dept_code)

        municipios = dept_data.get("municipios", {})
        for muni_code, muni_data in municipios.items():
            muni_norm = normalize_code(muni_code)

            for zona_code, zona_data in muni_data.get("zonas", {}).items():
                zona_norm = normalize_code(zona_code)

                for puesto_code, puesto_data in zona_data.get("puestos", {}).items():
                    puesto_norm = normalize_code(puesto_code)

                    key = (dept_norm, muni_norm, zona_norm, puesto_norm)
                    if key not in seletion:
                        continue  # Filtro llevado hasta el puesto: ni se pide el JSON de mesas

                    visited_keys.add(key)
                    tables_needed = seletion[key]

                    path_key = (
                        f"data/esc/v1/actas-documentos/{corporacion}/"
                        f"{dept_code}/{muni_code}/{zona_code}/{puesto_code}/mesas/"
                    )

                    if path_key not in index_map:
                        positions_not_found.append(key)
                        for missing_table in tables_needed:
                            meta = (dept_code, muni_code, zona_code, puesto_code, str(missing_table).zfill(3))
                            log_error(meta, "Puesto_no_encontrado_en_indice", path_key)
                        continue

                    names_json_tables = index_map[path_key]
                    url_json_tables = f"{BASE_URL}{path_key}{names_json_tables}"

                    print(
                        f"[*] Puesto seleccionado: {dept_code}/{muni_code}/{zona_code}/{puesto_code} "
                        f"({len(tables_needed)} mesa(s))"
                    )

                    tables = fetch_cached_json(url_json_tables, names_json_tables)
                    if not tables:
                        for missing_table in tables_needed:
                            meta = (dept_code, muni_code, zona_code, puesto_code, str(missing_table).zfill(3))
                            log_error(meta, "Fallo_descarga_JSON_mesas", url_json_tables)
                        continue

                    destination_path = os.path.join(BASE_DIR, dept_code, muni_code, zona_code)
                    os.makedirs(destination_path, exist_ok=True)

                    tables_found = set()
                    for mesa in tables:
                        try:
                            table_number = int(mesa.get("numero"))
                        except (TypeError, ValueError):
                            continue

                        if table_number not in tables_needed:
                            continue  # Filtro a nivel de mesa individual

                        relative_path_pdf = mesa.get("nombre_archivo")
                        if not relative_path_pdf:
                            tables_found.add(table_number)  # existe en el JSON, pero sin PDF asociado
                            meta = (dept_code, muni_code, zona_code, puesto_code, str(table_number).zfill(3))
                            log_error(meta, "Mesa_sin_archivo_pdf_asociado", url_json_tables)
                            continue

                        tables_found.add(table_number)

                        num_mesa_str = str(table_number).zfill(3)
                        final_pdf_url = f"{BASE_URL}{relative_path_pdf.lstrip('/')}"
                        filename = f"{dept_code}_{muni_code}_{zona_code}_{puesto_code}_{num_mesa_str}.pdf"
                        destination_file = os.path.join(destination_path, filename)

                        id = f"{dept_code}|{muni_code} - Puesto {puesto_code} - Mesa {num_mesa_str}"
                        meta = (dept_code, muni_code, zona_code, puesto_code, num_mesa_str)

                        pdf_tasks.append((final_pdf_url, destination_file, id, meta))

                    remaining = tables_needed - tables_found
                    for remaining_table in remaining:
                        meta = (dept_code, muni_code, zona_code, puesto_code, str(remaining_table).zfill(3))
                        log_error(meta, "Mesa_no_encontrada_en_JSON", url_json_tables)

    if positions_not_found:
        print(f"\n[!] {len(positions_not_found)} puesto(s) del CSV no existen en el índice de la Registraduría:")
        for key in positions_not_found:
            print(f"    - dd={key[0]} mm={key[1]} zz={key[2]} pp={key[3]}")

    unvisited_keys = set(seletion.keys()) - visited_keys
    if unvisited_keys:
        print(
            f"\n[!] {len(unvisited_keys)} puesto(s) del CSV NO existen como nodo en el árbol "
            f"del divipole.json (nunca se visitaron al recorrer departamentos/municipios/zonas/puestos):"
        )
        for k in unvisited_keys:
            print(f"    - dd={k[0]} mm={k[1]} zz={k[2]} pp={k[3]}")
            for m_table in seletion[k]:
                meta = (k[0], k[1], k[2], k[3], str(m_table).zfill(3))
                log_error(meta, "Puesto_no_encontrado_en_estructura_divipole", "")

    if not pdf_tasks:
        print("\n[-] Cero tareas programadas. Revisa el CSV de selección y los códigos.")
        return

    print(f"\n[=== FASE 3: DESCARGA DE PDFS ({len(pdf_tasks)} DOCUMENTOS) ===]")

    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
        next = [
            executor.submit(download_pdf, url, destino, id_mesa, meta)
            for url, destino, id_mesa, meta in pdf_tasks
        ]
        for n in as_completed(next):
            print(n.result())

    total_requested = sum(len(v) for v in seletion.values())
    missing = total_requested - len(pdf_tasks)
    print(f"\n[OK] Proceso terminado. {len(pdf_tasks)} mesa(s) encoladas de las {total_requested} solicitadas.")
    if missing > 0:
        print(f"[!] {missing} mesa(s) no se encolaron. Revisa {ERRORS_CSV_PATH} para ver el motivo exacto de cada una.")

