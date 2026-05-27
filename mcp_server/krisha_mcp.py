"""
MCP сервер для KrishaAgent.
Запуск: fastmcp run mcp_server/krisha_mcp.py

Предоставляет 3 инструмента:
1. search_listings     — поиск объявлений по параметрам
2. get_price_statistics — статистика цен по району
3. estimate_fair_value  — оценка справедливой стоимости
"""

import sys
import os
import json
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastmcp import FastMCP
import config

mcp = FastMCP("KrishaAgent MCP Server")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://krisha.kz/",
}

# Корректировки цены по этажу (эмпирические для рынка Алматы)
FLOOR_ADJUSTMENTS = {
    "first": -0.05,   # первый этаж: -5%
    "last": -0.03,    # последний этаж: -3%
    "middle": 0.0,    # средние этажи: без корректировки
}

CONDITION_ADJUSTMENTS = {
    "евроремонт": 0.10,
    "хороший": 0.03,
    "средний": 0.0,
    "требует ремонта": -0.12,
    "черновая": -0.20,
}


@mcp.tool()
def search_listings(
    district: str = "",
    rooms: int = 0,
    price_min: int = 0,
    price_max: int = 0,
    limit: int = 10,
) -> str:
    """
    Ищет объявления о продаже квартир на Krisha.kz.

    Args:
        district: Район Алматы (например "Бостандыкский", "Алмалинский")
        rooms: Количество комнат (0 = любое)
        price_min: Минимальная цена в тенге (0 = без ограничения)
        price_max: Максимальная цена в тенге (0 = без ограничения)
        limit: Количество результатов (макс 20)

    Returns:
        JSON-строка со списком объявлений
    """
    params = {"das[_sys.hasPhoto]": 1, "limit": min(limit, 20), "page": 1}

    if rooms:
        params["das[live.rooms]"] = rooms
    if price_min:
        params["das[price][from]"] = price_min
    if price_max:
        params["das[price][to]"] = price_max

    try:
        resp = requests.get(
            config.KRISHA_SEARCH_URL, params=params, headers=HEADERS, timeout=10
        )
        resp.raise_for_status()
        offers = resp.json().get("offers", {}).get("list", [])
    except Exception as e:
        return json.dumps({"error": str(e), "listings": []}, ensure_ascii=False)

    listings = []
    for offer in offers[:limit]:
        price = offer.get("price", 0)
        area = offer.get("square", 0) or 1
        listings.append({
            "id": offer.get("id"),
            "title": offer.get("title", ""),
            "price": price,
            "area": area,
            "price_m2": round(price / area) if area else 0,
            "floor": offer.get("floor", 0),
            "floors_total": offer.get("floorsTotal", 0),
            "district": offer.get("districtName", district),
            "url": f"https://krisha.kz/a/show/{offer.get('id', '')}",
        })

    return json.dumps({"listings": listings, "count": len(listings)}, ensure_ascii=False)


@mcp.tool()
def get_price_statistics(district: str = "", rooms: int = 0) -> str:
    """
    Возвращает статистику цен на квартиры по рынку Алматы.
    Использует исторические данные Krisha Analytics API.

    Args:
        district: Район Алматы (пустая строка = весь город)
        rooms: Количество комнат (0 = все типы)

    Returns:
        JSON с ценовой статистикой: avg_price_m2, median, trend и т.д.
    """
    try:
        params = {
            "id": 0,
            "rooms": rooms if rooms else "",
            "buildingType": "",
            "mode": "long",
            "geo": 2,  # Алматы
        }
        resp = requests.get(
            config.KRISHA_ANALYTICS_URL, params=params, headers=HEADERS, timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        # Берём последние 24 точки (~6 месяцев)
        series = data.get("data", {}).get("series", [])
        if not series:
            raise ValueError("нет данных")

        recent = series[-24:] if len(series) >= 24 else series
        prices = [p["value"] for p in recent if p.get("value")]

        if not prices:
            raise ValueError("нет цен")

        sorted_prices = sorted(prices)
        avg = sum(prices) / len(prices)
        median = sorted_prices[len(sorted_prices) // 2]

        # Тренд: сравниваем последние 4 недели с предыдущими 4
        trend_pct = 0.0
        if len(prices) >= 8:
            recent_avg = sum(prices[-4:]) / 4
            prev_avg = sum(prices[-8:-4]) / 4
            trend_pct = round((recent_avg - prev_avg) / prev_avg * 100, 2) if prev_avg else 0

        last_price = prices[-1]

        stats = {
            "district": district or "Алматы (весь город)",
            "rooms": rooms or "все",
            "avg_price_m2": round(avg),
            "median_price_m2": round(median),
            "min_price_m2": round(sorted_prices[0]),
            "max_price_m2": round(sorted_prices[-1]),
            "current_price_m2": round(last_price),
            "trend_4w_pct": trend_pct,
            "data_points": len(prices),
        }
        return json.dumps(stats, ensure_ascii=False)

    except Exception as e:
        # Fallback: актуальные данные на момент разработки
        fallback = {
            "district": district or "Алматы",
            "rooms": rooms or "все",
            "avg_price_m2": 800_000,
            "median_price_m2": 785_000,
            "min_price_m2": 450_000,
            "max_price_m2": 1_500_000,
            "current_price_m2": 800_445,
            "trend_4w_pct": 0.3,
            "note": f"fallback данные, причина: {str(e)}",
        }
        return json.dumps(fallback, ensure_ascii=False)


@mcp.tool()
def estimate_fair_value(
    area_sqm: float,
    district: str = "Алматы",
    floor: int = 5,
    floors_total: int = 10,
    condition: str = "средний",
) -> str:
    """
    Оценивает справедливую рыночную стоимость квартиры.

    Args:
        area_sqm: Площадь квартиры в м²
        district: Район Алматы
        floor: Этаж квартиры
        floors_total: Всего этажей в доме
        condition: Состояние ремонта: "евроремонт" / "хороший" / "средний" / "требует ремонта" / "черновая"

    Returns:
        JSON с оценкой: fair_min, fair_max, price_per_m2, verdict_guide
    """
    # Получаем базовую цену м² из API
    stats_json = get_price_statistics(district=district)
    stats = json.loads(stats_json)
    base_price_m2 = stats.get("current_price_m2", 800_000)

    # Корректировка на этаж
    if floor == 1:
        floor_adj = FLOOR_ADJUSTMENTS["first"]
    elif floor == floors_total:
        floor_adj = FLOOR_ADJUSTMENTS["last"]
    else:
        floor_adj = FLOOR_ADJUSTMENTS["middle"]

    # Корректировка на состояние
    condition_adj = CONDITION_ADJUSTMENTS.get(condition.lower(), 0.0)

    # Итоговая цена м²
    adjusted_m2 = base_price_m2 * (1 + floor_adj + condition_adj)
    fair_price = adjusted_m2 * area_sqm

    result = {
        "district": district,
        "area_sqm": area_sqm,
        "base_price_m2": round(base_price_m2),
        "adjusted_price_m2": round(adjusted_m2),
        "floor_adjustment_pct": round(floor_adj * 100, 1),
        "condition_adjustment_pct": round(condition_adj * 100, 1),
        "fair_price_min": round(fair_price * 0.92),
        "fair_price_max": round(fair_price * 1.08),
        "fair_price_mid": round(fair_price),
        "verdict_guide": (
            "выгодно если цена < {:,}₸ | "
            "рыночная {:,}₸–{:,}₸ | "
            "дорого если > {:,}₸"
        ).format(
            round(fair_price * 0.92),
            round(fair_price * 0.92),
            round(fair_price * 1.08),
            round(fair_price * 1.08),
        ),
    }
    return json.dumps(result, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()
