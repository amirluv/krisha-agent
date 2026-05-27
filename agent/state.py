from typing import TypedDict, Optional


class AgentState(TypedDict):
    # Входные данные
    listing_url: str              # URL объявления на Krisha.kz

    # Данные объявления (спарсены со страницы)
    listing_data: Optional[dict]  # {title, price, area, floor, floors_total, district, rooms, photo_urls, description}

    # Найденные похожие объявления для сравнения
    similar_listings: list

    # Статистика по рынку
    price_stats: dict             # {avg_price_m2, median, min, max, trend_6m}

    # Анализ фото (GPT-4o Vision, автоматически из объявления)
    photo_analysis: Optional[dict]  # {condition_score, layout, issues, price_adjustment_pct, summary}

    # Оценка справедливой цены
    price_evaluation: Optional[dict]  # {fair_min, fair_max, verdict, listing_price}

    # Финальная рекомендация
    recommendation: Optional[str]

    # Human-in-the-loop (уточняющий вопрос после рекомендации)
    user_feedback: Optional[str]
    awaiting_feedback: bool

    # Служебное
    iteration: int
    error: Optional[str]
