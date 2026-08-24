import sys
from src.ingestion.extract_e14_sample import generate_e14_sample
from src.scraper.scraper_e14 import execute_scraper


def main():
    print("[1/2] Iniciando cálculo y generación de muestra E-14...")
    df_sample = generate_e14_sample()
    print(f"Muestra generada: {len(df_sample)} registros.")

    print("[2/2] Iniciando descarga de formularios con scraper...")
    execute_scraper()
    print("Pipeline finalizado con éxito.")


if __name__ == "__main__":
    main()