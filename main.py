import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Procesa los argumentos de línea de comandos para controlar qué fase ejecutar.

    Returns:
        argparse.Namespace: Argumentos validados (`phase` y `pdf_dir`).
    """
    parser = argparse.ArgumentParser(
        description=(
            "E-14 Document AI & ETL Pipeline "
            "(Fase 1: Muestreo/Scraping | Fase 2: Visión Artificial/OCR/BD)"
        )
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["1", "2", "all"],
        default="all",
        help="Fase a ejecutar: '1' (Muestreo + Scraper), '2' (Visión + OCR + BD), 'all' (Ambas).",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="Directorio raíz con PDFs E-14 para la Fase 2 (sobreescribe E14_PDF_DIR y data/).",
    )
    return parser.parse_args()


def main() -> None:
    """Punto de entrada principal para la ejecución modular o end-to-end del pipeline E-14."""
    args = parse_args()

    # Fase 1: Ingestión DIVIPOLE, muestreo estratificado y descarga concurrente de actas PDF
    if args.phase in ("1", "all"):
        from src.ingestion.extract_e14_sample import generate_e14_sample
        from src.scraper.scraper_e14 import execute_scraper

        print("[FASE 1 - 1/2] Iniciando cálculo y generación de muestra E-14...")
        df_sample = generate_e14_sample()
        print(f"Muestra generada: {len(df_sample)} registros.")

        print("[FASE 1 - 2/2] Iniciando descarga de formularios con scraper...")
        execute_scraper()

    # Fase 2: Alineación geométrica, OCR/HTR/OMR y carga atómica ACID en Supabase
    if args.phase in ("2", "all"):
        from src.processing.pipeline import execute_ocr_pipeline

        print("[FASE 2] Iniciando procesamiento de visión artificial, OCR/HTR y carga a Supabase...")
        pdf_dir = Path(args.pdf_dir) if args.pdf_dir else None
        execute_ocr_pipeline(pdf_root_dir=pdf_dir)

    print("Ejecución finalizada con éxito.")


if __name__ == "__main__":
    main()