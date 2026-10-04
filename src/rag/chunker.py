import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cti"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "documents"
    / "chunks.json"
)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def clean_text(text):
    text = text.replace("\x00", " ")

    text = re.sub(
        r"\r\n?",
        "\n",
        text
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def split_into_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(
            start + chunk_size,
            len(words)
        )

        chunk_words = words[start:end]

        chunk_text = " ".join(
            chunk_words
        )

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": chunk_text,
                "start_word": start,
                "end_word": end
            }
        )

        if end >= len(words):
            break

        start = end - overlap
        chunk_id += 1

    return chunks


def process_documents():
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"CTI directory not found: {INPUT_DIR}"
        )

    documents = []

    files = sorted(
        INPUT_DIR.glob("*.txt")
    )

    if not files:
        raise RuntimeError(
            f"No .txt CTI reports found in: {INPUT_DIR}"
        )

    global_chunk_id = 0

    for file_path in files:
        print(
            f"Processing: {file_path.name}"
        )

        text = file_path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

        text = clean_text(text)

        chunks = split_into_chunks(
            text
        )

        for chunk in chunks:
            documents.append(
                {
                    "chunk_id": global_chunk_id,
                    "document_id": file_path.stem,
                    "document_name": file_path.name,
                    "source_path": str(
                        file_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),
                    "local_chunk_id": chunk[
                        "chunk_id"
                    ],
                    "start_word": chunk[
                        "start_word"
                    ],
                    "end_word": chunk[
                        "end_word"
                    ],
                    "text": chunk["text"]
                }
            )

            global_chunk_id += 1

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            documents,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 50)
    print("       CTI DOCUMENT CHUNKING")
    print("=" * 50)
    print()

    print(
        f"Documents processed: {len(files)}"
    )

    print(
        f"Total chunks: {len(documents)}"
    )

    print(
        f"Chunk size: {CHUNK_SIZE} words"
    )

    print(
        f"Chunk overlap: {CHUNK_OVERLAP} words"
    )

    print()
    print("Saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    process_documents()