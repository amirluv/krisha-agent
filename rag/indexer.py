"""
Загружает реальные объявления из SQLite в ChromaDB для RAG-поиска.
Запуск: python rag/indexer.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
from chromadb.utils import embedding_functions
import config
from data.database import get_conn, get_total_count

BATCH_SIZE = 500


def listing_to_text(row: dict) -> str:
    return (
        f"{row['rooms']}-комнатная квартира, {row['district']} район, "
        f"{row['area']}м², {row['floor']}/{row['floors_total']} этаж, "
        f"цена {row['price']:,}₸, {int(row['price_m2']):,}₸/м²"
    )


def index_from_sqlite():
    total = get_total_count()
    if total == 0:
        print("База данных пустая — сначала запусти scraper.py")
        return

    print(f"Найдено {total:,} объявлений в SQLite. Начинаю индексацию...\n")

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY,
        model_name=config.MODEL_EMBEDDINGS,
    )

    try:
        client.delete_collection(config.COLLECTION_NAME)
        print("Старая коллекция удалена.")
    except Exception:
        pass

    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=ef,
        metadata={"description": "Krisha.kz listings Almaty"},
    )

    conn = get_conn()
    offset = 0
    indexed = 0

    while True:
        rows = conn.execute(
            "SELECT * FROM listings WHERE price_m2 > 0 LIMIT ? OFFSET ?",
            (BATCH_SIZE, offset),
        ).fetchall()

        if not rows:
            break

        documents = []
        metadatas = []
        ids = []

        for row in rows:
            row = dict(row)
            documents.append(listing_to_text(row))
            metadatas.append({
                "type":         "listing",
                "listing_id":   str(row["listing_id"]),
                "district":     row["district"],
                "rooms":        row["rooms"],
                "area":         float(row["area"]),
                "floor":        row["floor"],
                "floors_total": row["floors_total"],
                "price":        row["price"],
                "price_m2":     float(row["price_m2"]),
            })
            ids.append(f"listing_{row['listing_id']}")

        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        indexed += len(rows)
        offset += BATCH_SIZE
        print(f"  Проиндексировано: {indexed:,} / {total:,}")

    conn.close()
    print(f"\n✓ Готово! В ChromaDB загружено {collection.count():,} объявлений.")


if __name__ == "__main__":
    index_from_sqlite()
