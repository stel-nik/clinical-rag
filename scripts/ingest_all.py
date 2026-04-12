import httpx
from pathlib import Path

SAMPLES_DIR = Path("data/samples")
API_URL = "http://localhost:8000"

def ingest_file(filepath: Path):
    with open(filepath, "rb") as f:
        upload = (filepath.name, f, "text/plain")
        response = httpx.post(
            f"{API_URL}/documents/ingest",
            files={"file": upload}
        )
    return response.json()

def main():
    txt_files = list(SAMPLES_DIR.glob("*.txt"))

    if not txt_files:
        print("No .txt files found in", SAMPLES_DIR)
        return

    print(f"Found {len(txt_files)} files to ingest...")

    for filepath in txt_files:
        print(f"Ingesting {filepath.name}")
        result = ingest_file(filepath)
        print(f"Done: {result}")

if __name__ == "__main__":
    main()