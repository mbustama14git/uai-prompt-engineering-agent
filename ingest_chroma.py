from pathlib import Path
from bd.chroma_store import upsert_docs, count, debug_collections, CHROMA_DIR

DATA_DIR = Path("data_txt")

def main():
    print("Usando CHROMA_DIR:", CHROMA_DIR)
    print("Colecciones antes:", debug_collections())
    print("Count antes:", count())

    paths = sorted(DATA_DIR.glob("*.txt"))
    if not paths:
        raise SystemExit(f"No hay .txt en: {DATA_DIR.resolve()}")

    ids = [p.stem for p in paths]
    texts = [p.read_text(encoding="utf-8", errors="ignore") for p in paths]
    metas = [{"source": str(p)} for p in paths]

    upsert_docs(ids, texts, metas)

    print("Colecciones después:", debug_collections())
    print("Count después:", count())

if __name__ == "__main__":
    main()
