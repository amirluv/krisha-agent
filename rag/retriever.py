import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
from chromadb.utils import embedding_functions
import config

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is not None:
        return _collection

    _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=config.OPENAI_API_KEY,
        model_name=config.MODEL_EMBEDDINGS,
    )
    try:
        _collection = _client.get_collection(
            name=config.COLLECTION_NAME,
            embedding_function=ef,
        )
    except Exception:
        # Коллекция ещё не создана — запусти rag/indexer.py
        _collection = None
    return _collection


def retrieve_similar_listings(query: str, top_k: int = 5) -> list[dict]:
    """
    Ищет похожие объявления в ChromaDB по текстовому запросу.
    Возвращает список объявлений в формате совместимом с nodes.py.
    """
    collection = _get_collection()
    if collection is None:
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            where={"type": "listing"},  # только объявления, не price_history
        )
    except Exception:
        return []

    listings = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    for doc, meta in zip(documents, metadatas):
        listings.append({
            "title": doc,
            "price": meta.get("price", 0),
            "area": meta.get("area", 0),
            "floor": meta.get("floor", 0),
            "floors_total": 0,
            "district": meta.get("district", ""),
            "url": "",
            "price_m2": meta.get("price_m2", 0),
            "source": "rag",
        })

    return listings


def retrieve_price_history(query: str = "цена м²", top_k: int = 10) -> list[dict]:
    """Возвращает исторические ценовые данные из ChromaDB."""
    collection = _get_collection()
    if collection is None:
        return []

    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            where={"type": "price_history"},
        )
    except Exception:
        return []

    history = []
    metadatas = results.get("metadatas", [[]])[0]
    for meta in metadatas:
        history.append({
            "date": meta.get("date", ""),
            "price_m2": meta.get("price_m2", 0),
        })

    return sorted(history, key=lambda x: x["date"])
