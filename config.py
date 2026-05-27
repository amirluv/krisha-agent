import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys (локально — из .env, на Streamlit Cloud — из secrets) ---
def _get_secret(key: str) -> str:
    try:
        import streamlit as st
        return st.secrets.get(key, os.getenv(key, ""))
    except Exception:
        return os.getenv(key, "")

OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
LANGSMITH_API_KEY = _get_secret("LANGSMITH_API_KEY")

# --- LangSmith ---
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "krisha-agent")
if LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY

# --- Models ---
# text: дёшево и быстро для поиска / парсинга / оценки
MODEL_TEXT = "gpt-4o-mini"
# vision + финальная рекомендация: нужно качество
MODEL_SMART = "gpt-4o"
# embeddings
MODEL_EMBEDDINGS = "text-embedding-3-small"

# --- Hyperparameters (documented for defence) ---
# parse_query: детерминированность критична — извлекаем структурированные данные
TEMP_PARSE = 0.0
# search / evaluate: минимальная вариативность
TEMP_ANALYZE = 0.1
# recommendation: нужен связный нарратив
TEMP_RECOMMEND = 0.4

MAX_TOKENS_PARSE = 300
MAX_TOKENS_ANALYZE = 600
MAX_TOKENS_RECOMMEND = 800

TOP_P = 0.9

# --- RAG ---
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "krisha_listings"
RAG_TOP_K = 5

# --- Krisha API ---
KRISHA_ANALYTICS_URL = "https://krisha.kz/spa-api/content/analytics/sale"
KRISHA_SEARCH_URL = "https://krisha.kz/a/api/offers"

# --- Agent ---
MAX_SEARCH_ITERATIONS = 3   # защита от бесконечного цикла
