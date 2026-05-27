# KrishaAgent — Архитектура

## Обзор

KrishaAgent — AI-агент для анализа объявлений о продаже квартир на Krisha.kz. Агент помогает покупателю понять: справедлива ли цена, стоит ли торговаться, есть ли лучшие варианты на рынке.

```
Пользователь → [URL объявления] → KrishaAgent → [Вердикт + Рекомендация]
```

---

## Архитектурная диаграмма

```
┌─────────────────────────────────────────────────────────────────┐
│                        ВХОДНЫЕ ДАННЫЕ                           │
│              URL объявления (krisha.kz/a/show/...)              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LANGGRAPH ГРАФ (6 НОД)                       │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ fetch_listing│───▶│search_similar│───▶│  analyze_photos  │  │
│  │              │    │              │    │                  │  │
│  │ BeautifulSoup│    │ ChromaDB RAG │    │ GPT-4o Vision    │  │
│  │ regex парсинг│    │ top-5 похожих│    │ все 10 фото      │  │
│  └──────┬───────┘    └──────────────┘    └────────┬─────────┘  │
│         │ (error → END)                           │            │
│         │                                         ▼            │
│         │            ┌──────────────┐    ┌──────────────────┐  │
│         │            │  human_review│◀───│evaluate_price    │  │
│         │            │              │    │                  │  │
│         │            │ follow-up Q&A│    │ SQLite + IQR     │  │
│         │            │ GPT-4o-mini  │    │ корректировки    │  │
│         │            └──────────────┘    └────────┬─────────┘  │
│         │                                         │            │
│         │                            ┌────────────▼──────────┐ │
│         │                            │ generate_recommendation│ │
│         │                            │                        │ │
│         │                            │ GPT-4o, 5 разделов    │ │
│         │                            └────────────────────────┘ │
└─────────┼───────────────────────────────────────────────────────┘
          │
          ▼ error path
         END
```

---

## Компоненты системы

### 1. LangGraph граф (`agent/`)

Оркестрация пайплайна через StateGraph с ветвлением.

```python
# Ветвление после fetch_listing:
fetch_listing ──→ ok ──→ search_similar ──→ analyze_photos
              └──→ error ──→ END
```

**State** (`agent/state.py`) — TypedDict с 10 полями, передаётся между нодами.

### 2. MCP Сервер (`mcp_server/krisha_mcp.py`)

FastMCP-сервер с 3 инструментами:

| Tool | Описание | Источник данных |
|------|---------|----------------|
| `search_listings` | Поиск квартир по параметрам | Krisha API |
| `get_price_statistics` | Статистика цен м² | Krisha Analytics API |
| `estimate_fair_value` | Оценка справедливой цены | Расчёт с корректировками |

### 3. RAG Пайплайн (`rag/`)

```
SQLite (37 763 объявл.) ──→ indexer.py ──→ ChromaDB
                                          (36 686 эмбеддингов)
                                          text-embedding-3-small
                                                  │
                                                  ▼
                                         retriever.py
                                         top-5 по косинусному
                                         сходству
```

**Почему ChromaDB:** локальная база, нет задержек сети, бесплатно, 36k документов умещаются в памяти.  
**Почему text-embedding-3-small:** лучший баланс качества/стоимости для русскоязычных текстов о недвижимости.

### 4. База данных (`data/`)

```
SQLite: krisha_listings.db
├── 37 763 объявления
├── 8 районов Алматы  
├── 1–5 комнат + без фильтра
└── Индекс: (district, rooms)

Поля: listing_id, district, rooms, area, floor, 
      floors_total, price, price_m2, scraped_at
```

**IQR фильтрация выбросов:** медиана устойчива к аномальным ценам (нотариальные, подарочные сделки).  
**Площадь ±30%:** 60м² квартира сравнивается только с 42–78м², не со всеми 2-комнатными.

### 5. Ценовой движок (`agent/nodes.py: evaluate_price`)

```
avg_price_m2 (из SQLite)
    × (1 + корректировки)
    ──────────────────────
    Корректировки:
    • Этаж:    1-й = -5%, последний = -3%
    • Год:     <1950 = -35%, ..., ≥2020 = +8%
    • Тип дома: панель = -10%, монолит = +5%
    • Фото:    score 9-10 = +10%, ..., 1-2 = -15%
    ──────────────────────
    fair_price = fair_m2 × area
    диапазон:  [fair_price × 0.92, fair_price × 1.08]
```

### 6. Vision анализ (`agent/nodes.py: analyze_photos`)

- Скачивает все фото из объявления (до 10)
- GPT-4o получает все фото в одном запросе
- Строгий промпт с якорями оценки: 1-2 аварийное / 3-4 требует полного / 5-6 косметика / 7-8 свежий / 9-10 евро
- Корректировка цены рассчитывается детерминированно по score, не моделью

### 7. Мониторинг (LangSmith)

```python
# config.py
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "krisha-agent"
```

Каждый запуск `run_agent()` → отдельный trace в LangSmith с историей нод, токенами, latency.

---

## Стек технологий

| Слой | Технология | Обоснование |
|------|-----------|-------------|
| Оркестрация | LangGraph 0.1 | Граф с ветвлениями и human-in-the-loop |
| Frontend | Streamlit | Быстрый прототип без backend |
| LLM | GPT-4o / GPT-4o-mini | Vision нужен o-4, текст — mini достаточно |
| Embeddings | text-embedding-3-small | Лучший баланс цены/качества |
| Vector DB | ChromaDB | Локальная, нет latency сети |
| SQL | SQLite | Для 37k строк — достаточно, нет overhead |
| MCP | FastMCP | Минимум бойлерплейта |
| Scraper | requests + BeautifulSoup | Простота, стабильность |
| Мониторинг | LangSmith | Обязательно по ТЗ, бесплатный тир |

---

## Структура проекта

```
krisha-agent/
├── app.py                    # Streamlit UI
├── config.py                 # Модели, температуры, пути
├── agent/
│   ├── graph.py              # LangGraph граф + run_agent()
│   ├── nodes.py              # 6 нод агента
│   └── state.py              # AgentState TypedDict
├── mcp_server/
│   └── krisha_mcp.py         # FastMCP сервер (3 tools)
├── rag/
│   ├── indexer.py            # Индексация SQLite → ChromaDB
│   └── retriever.py          # Поиск похожих (top-5)
├── data/
│   ├── database.py           # SQLite: get_price_stats, upsert
│   ├── scraper.py            # Bulk scraper с checkpoint
│   └── krisha_listings.db    # 37 763 объявления
├── chroma_db/                # ChromaDB (36 686 эмбеддингов)
├── skills/
│   └── krisha_analyzer/
│       └── SKILL.md          # Описание агента
├── evals/
│   ├── golden_dataset.json   # 30 тестовых примеров
│   ├── run_evals.py          # Автоматический прогон
│   ├── results_config_a.json # Результаты Config A
│   └── results_config_b.json # Результаты Config B
├── ARCHITECTURE.md           # Этот файл
├── EVALS.md                  # Результаты эвалюаций
└── README.md                 # Инструкция запуска
```

---

## Поток данных (один запрос)

```
1. Пользователь вставляет URL
   ↓
2. fetch_listing: скрапим krisha.kz/a/show/<id>
   → заголовок, цена, площадь, этаж, район, фото
   ↓
3. search_similar: ChromaDB top-5 похожих
   → "2-комн Бостандыкский 65м²" → 5 объявлений из базы
   ↓
4. analyze_photos: все 10 фото → GPT-4o Vision
   → condition_score=7, renovation_needed="не нужен"
   ↓
5. evaluate_price: SQLite → avg_m2 по (district, rooms, area±30%)
   → IQR фильтрация → корректировки → вердикт
   ↓
6. generate_recommendation: GPT-4o
   → структурированный текст с цифрами
   ↓
7. human_review: ожидает follow-up вопрос
   → GPT-4o-mini отвечает в контексте
```

**Типичное время запроса:** 15–25 секунд  
(~8с скрапинг + ~5с Vision + ~3с рекомендация + ~2с RAG)
