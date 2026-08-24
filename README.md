# E-14 Document AI & ETL Pipeline | Fase 1: Ingestión, Muestreo Estratificado y Scraping Asíncrono

Pipeline modular de ingeniería de datos y diseño experimental desarrollado para la digitalización automatizada y extracción de datos mediante Visión por Computadora OCR (Optical Character Recognition) y HTR (Handwritten Text Recognition) de formularios electorales E-14 (Segunda Vuelta Presidencial 2026, Colombia).

Este repositorio (por ahora) implementa la **Fase 1**: extracción de datos maestros desde PostgreSQL, muestreo probabilístico estratificado ($n = 383$) y un motor de web scraping concurrente y dirigido.

---

## 1. Arquitectura del Flujo de Datos

## 1. Arquitectura del Flujo de Datos

## 1. Arquitectura del Flujo de Datos

```mermaid
flowchart TD
    subgraph ORQ [" Orquestador Principal (main.py) "]
        direction TB
        M["🚀 Inicio del Pipeline<br><code>main.py</code>"]
    end

    subgraph Script1 [" Ingestión y Muestreo (src/ingestion/extract_e14_sample.py) "]
        direction TB
        SB[("🗄️ Supabase (PostgreSQL)<br>Tabla: divipole_regis")] --> EXT["⚙️ Extracción Paginada<br>& Filtro Consular (dd != 88)"]
        EXT --> STRAT["📐 Estratificación (zz)<br>Urbano / Rural / Cárcel"]
        STRAT --> COCH["🧮 Tamaño Muestral<br>(Cochran) 95% Confianza<br>5% Error"]
        COCH --> EXP["🔀 Expansión a Nivel Mesa<br>& Muestreo Aleatorio<br>(Seed=42)"]
        EXP --> CSV1["📊 Directorio de Datos<br><code>data/muestra_e14_segunda<br>_vuelta.csv</code> (n=383)"]
    end

    subgraph Script2 [" Scraping y Descarga Concurrente (src/scraper/scraper_e14.py) "]
        direction TB
        API["🌐 API Registraduría 2026<br>(index.json / divipole.json)"] <--> CACHE[("💾 Caché Local<br><code>data/.cache/</code>")]
        CSV1 --> LOAD["📥 Carga de Selección<br>& Normalización de Códigos"]
        CACHE --> LOAD
        LOAD --> RESOLV["🔍 Resolución Jerárquica<br>Depto ➔ Muni ➔ Zona ➔<br>Puesto ➔ Mesa"]
        RESOLV --> POOL["⚡ ThreadPoolExecutor<br>(curl_cffi + Impersonación<br>Chrome)"]
        
        POOL --> PDF["📑 Almacenamiento Estructurado<br><code>data/{dd}/{mm}/{zz}/<br>{dd}_{mm}_{zz}_{pp}_{mesa}.pdf</code>"]
        POOL --> ERR["📝 Auditoría de Errores<br>(Thread-Safe Lock)<br><code>data/errores_descarga.csv</code>"]
    end

    ORQ --> Script1
    Script1 --> Script2
```

---

## 2. Componentes del Sistema

### Módulo de Ingestión y Muestreo (`src/ingestion/`)
* **Persistencia Relacional:** Conexión paginada contra **Supabase (PostgreSQL)** sobre la tabla `divipole_regis`.
* **Delimitación del Universo:** Filtrado estricto de mesas que no se encontraron en (https://escrutinios2vueltapresidente2026.registraduria.gov.co/actas-e14) para este caso consulares (`dd == '88'`), esto con el fin de garantizar la trazabilidad del dataset frente a datos de auditoría oficiales.
  * **Puestos procesados:** 13.489
  * **Universo estadístico ($N$):** 118.346 mesas.
* **Diseño Experimental (Fórmula de Cochran):**
  Cálculo del tamaño muestral para poblaciones finitas bajo varianza máxima ($p = q = 0.5$):

$$n = \frac{N \cdot Z^2 \cdot p \cdot q}{e^2 \cdot (N - 1) + Z^2 \cdot p \cdot q}$$

  * $Z = 1.96$ ($95\%$ Confianza)
  * $e = 0.05$ ($5\%$ Margen de Error)
  * **Tamaño muestral resultante:** $n = 383\text{ mesas}$

* **Estratificación Proporcional:**
  Distribución controlada por código de zona (`zz`):

| Estrato Operativo | Código Zona (`zz`) | Total Mesas ($N_i$) | Representatividad | Muestra ($n_i$) |
| :--- | :--- | :--- | :--- | :--- |
| **Urbano** | `00` a `97` | 98.819 | 83.50% | **319** |
| **Rural** | `99` | 19.365 | 16.36% | **63** |
| **Penitenciario** | `98` | 162 | 0.14% | **1** |
| **Total** | — | **118.346** | **100.0%** | **383** |

* **Extracción Aleatoria:** Muestreo Aleatorio Simple (MAS) sin reemplazo dentro de cada estrato fijando semilla (`SEED=42`) para garantizar reproducibilidad.

---

### Módulo de Scraping Asíncrono Dirigido (`src/scraper/`)
* **Extracción Dirigida:** Ingestión exclusiva de las actas requeridas en `muestra_e14_segunda_vuelta.csv`, evitando transferencias masivas redundantes.
* **Resiliencia de Red y Evasión TLS:** Uso de `curl_cffi` con impersonación de huella digital del navegador (`chrome110`) y *jitter* aleatorio para evitar bloqueos por tasa de peticiones (*rate limiting*).
* **Caché Estructural:** Almacenamiento local de esquemas JSON intermedios (`.cache/`) para permitir reejecuciones idempotentes y descargas incrementales.
* **Validación de Integridad:** Comprobación binaria de cabeceras (`%PDF`) en la respuesta HTTP antes del guardado en disco.
* **Auditoría Thread-Safe:** Registro sincronizado con `threading.Lock` en `errores_descarga.csv` ante inconsistencias de índices o fallos de red.

---

## 3. Estructura del Repositorio

```text
e14-vision-ocr-pipeline/
├── data/
│   ├── divipole_2026.csv                   # Estructura maestra nacional
│   └── muestra_e14_segunda_vuelta.csv      # Muestra probabilística n=383
├── notebooks/
│   ├── e14_extraccion_muestra.ipynb        # EDA y experimentación interactiva
│   └── e14_vision_ocr_pipeline.ipynb       # (En curso) Pipeline de visión artificial y OCR
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── e14_extraccion_muestra.py       # Lógica de muestreo y conexión Supabase
│   ├── scraper/
│   │   ├── __init__.py
│   │   └── scraper_e14.py                  # Scraper asíncrono y auditoría
│   └── processing/
│       ├── __init__.py
│       ├── preprocess.py                   # (En curso)Homografía y rectificación
│       ├── tools.py                        # (En curso)Segmentación de ROI
│       ├── ocr.py                          # (En curso)Inferencia OCR / HTR
│       └── view.py                         # (En curso)Visualización de crops
├── .gitignore
├── main.py                                 # Ejecución de pipeline end-to-end
├── pyproject.toml
├── README.md
└── requirements.txt

4. Instalación y Ejecución
4.1. Clonar el repositorio y configurar el entorno

    git clone [https://github.com/tu-usuario/e14-vision-ocr-pipeline.git](https://github.com/tu-usuario/e14-vision-ocr-pipeline.git)
    cd e14-vision-ocr-pipeline
    python -m venv .venv
    source .venv/bin/activate   # En Windows: .venv\Scripts\activate
    pip install -r requirements.txt (Se cargaron dos archivos)
    pip install -e .

4.2. Configurar variables de entorno
Previamente cargada una tabla en supabase con el nombre de divipole_regis con 
la data que esta en "..\data\divipole_2026.csv"
Crea un archivo .env en la raíz a partir de la plantilla:
    
    SUPABASE_URL="[https://tu-proyecto.supabase.co](https://tu-proyecto.supabase.co)"
    SUPABASE_KEY="tu-key"

4.3. Ejecución del Pipeline
    Ejecución del pipeline completo de Fase 1 (Generación de muestra y descarga dirigida de PDFs):

    python main.py

