import json
import re
import base64
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from agent.state import AgentState
from rag.retriever import retrieve_similar_listings
from data.database import get_price_stats, get_total_count

client = OpenAI(api_key=config.OPENAI_API_KEY)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ---------------------------------------------------------------------------
# NODE 1: fetch_listing
# Скрапит страницу объявления и извлекает все данные
# ---------------------------------------------------------------------------
def fetch_listing(state: AgentState) -> AgentState:
    url = state["listing_url"].strip()

    # Нормализуем URL
    if not url.startswith("http"):
        url = "https://" + url

    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Заголовок: "3-комнатная квартира · 60 м² · 1/3 этаж, мкр Рахат"
        title_el = soup.find("h1")
        title = title_el.text.strip() if title_el else ""

        # Парсим из заголовка
        rooms = _parse_rooms(title)
        area = _parse_area(title)
        floor, floors_total = _parse_floor(title)

        # Цена
        price = _parse_price(soup)

        # Район
        district = _parse_district(soup)

        # Описание
        desc_el = soup.find(class_=re.compile(r"description|desc", re.I))
        description = desc_el.text.strip()[:500] if desc_el else ""

        # Фото — берём из JS в странице
        photo_urls = _parse_photos(resp.text)

        listing_data = {
            "title": title,
            "price": price,
            "area": area,
            "floor": floor,
            "floors_total": floors_total,
            "rooms": rooms,
            "district": district,
            "description": description,
            "photo_urls": photo_urls[:10],  # максимум 10 фото
            "url": url,
            "price_m2": round(price / area) if area else 0,
        }

        return {**state, "listing_data": listing_data, "error": None, "iteration": 0}

    except Exception as e:
        return {**state, "listing_data": None, "error": f"Не удалось загрузить объявление: {e}"}


def _parse_rooms(title: str) -> int:
    m = re.search(r"(\d+)-комнат", title)
    return int(m.group(1)) if m else 0


def _parse_area(title: str) -> float:
    m = re.search(r"([\d.]+)\s*м²", title)
    return float(m.group(1)) if m else 0.0


def _parse_floor(title: str) -> tuple[int, int]:
    m = re.search(r"(\d+)/(\d+)\s*этаж", title)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def _parse_price(soup: BeautifulSoup) -> int:
    for el in soup.find_all(class_=re.compile(r"price", re.I)):
        text = el.text.strip()
        digits = re.sub(r"[^\d]", "", text)
        if 6 <= len(digits) <= 12:
            return int(digits)
    return 0


def _parse_district(soup: BeautifulSoup) -> str:
    # Ищем "Алматы, Бостандыкский р-н" в параметрах
    for el in soup.find_all(class_=re.compile(r"param|detail|info|char", re.I)):
        text = el.text
        m = re.search(r"([А-ЯЁ][а-яё]+ский)\s+р", text)
        if m:
            return m.group(1)
    # Fallback: ищем в полном тексте
    m = re.search(r"([А-ЯЁ][а-яё]+ский)\s+р", soup.text)
    return m.group(1) if m else "Алматы"


def _parse_photos(html: str) -> list[str]:
    # Фото в JSON внутри скрипта
    urls = re.findall(r"https://krisha-photos[^\"']+\.jpg", html)
    # Убираем дубли (превьюшки vs full) — берём full
    full = [u for u in urls if "full" in u]
    return full if full else list(dict.fromkeys(urls))


def _parse_building_info(description: str) -> tuple[int, str]:
    """Извлекает год постройки и тип дома из описания объявления."""
    year = 0
    building_type = ""

    year_m = re.search(r"(?:Год постройки\s*\n+\s*|г\.?\s*п\.?)(\d{4})", description, re.I)
    if not year_m:
        year_m = re.search(r"(\d{4})\s*г\.?\s*п", description, re.I)
    if year_m:
        y = int(year_m.group(1))
        if 1900 <= y <= 2030:
            year = y

    bt_m = re.search(r"(монолитный|монолит|кирпичный|кирпич|панельный|панель)", description, re.I)
    if bt_m:
        raw = bt_m.group(1).lower()
        if "монол" in raw:
            building_type = "монолитный"
        elif "кирпич" in raw:
            building_type = "кирпичный"
        elif "панел" in raw:
            building_type = "панельный"

    return year, building_type


def _compute_structural_adjustments(floor: int, floors_total: int, year: int, building_type: str) -> dict:
    """
    Возвращает процентные корректировки к рыночной цене м².

    Источники коэффициентов: типичные дисконты на рынке Алматы.
    """
    floor_adj = 0.0
    if floor and floors_total:
        if floor == 1:
            floor_adj = -5.0
        elif floor == floors_total:
            floor_adj = -3.0

    year_adj = 0.0
    if year:
        if year >= 2020:
            year_adj = 8.0
        elif year >= 2010:
            year_adj = 2.0
        elif year >= 2000:
            year_adj = 0.0
        elif year >= 1990:
            year_adj = -5.0
        elif year >= 1970:
            year_adj = -15.0
        elif year >= 1960:
            year_adj = -20.0
        elif year >= 1950:
            year_adj = -28.0
        else:
            year_adj = -35.0

    bt_adj = 0.0
    bt_map = {"монолитный": 5.0, "кирпичный": 0.0, "панельный": -10.0}
    if building_type in bt_map:
        bt_adj = bt_map[building_type]

    total = floor_adj + year_adj + bt_adj
    return {
        "floor_adj_pct": floor_adj,
        "year_adj_pct": year_adj,
        "building_type_adj_pct": bt_adj,
        "total_structural_adj_pct": round(total, 1),
        "year": year,
        "building_type": building_type,
    }


# ---------------------------------------------------------------------------
# NODE 2: search_similar
# Ищет похожие объявления для сравнения с рынком
# ---------------------------------------------------------------------------
def search_similar(state: AgentState) -> AgentState:
    data = state.get("listing_data")
    if not data:
        return {**state, "similar_listings": []}

    # RAG поиск по параметрам объявления
    query = (
        f"{data['rooms']}-комнатная квартира {data['district']} район "
        f"{data['area']}м² {data['floor']} этаж"
    )
    rag_listings = retrieve_similar_listings(query, top_k=config.RAG_TOP_K)

    # Дополняем live данными из Krisha API
    live_listings = _fetch_similar_from_api(data)

    # Объединяем, дедупликация по url (live) или title (rag)
    seen = set()
    all_listings = []
    for lst in live_listings + rag_listings:
        url = lst.get("url", "")
        key = url if url else lst.get("title", "")
        if key and key not in seen and url != data.get("url", ""):
            seen.add(key)
            all_listings.append(lst)

    return {**state, "similar_listings": all_listings}


def _fetch_similar_from_api(data: dict) -> list:
    try:
        params = {
            "das[_sys.hasPhoto]": 1,
            "das[live.rooms]": data.get("rooms") or "",
            "limit": 10,
        }
        params = {k: v for k, v in params.items() if v != ""}
        resp = requests.get(
            config.KRISHA_SEARCH_URL, params=params, headers=HEADERS, timeout=10
        )
        offers = resp.json().get("offers", {}).get("list", [])
        result = []
        for o in offers[:8]:
            price = o.get("price", 0)
            area = o.get("square", 0) or 1
            result.append({
                "title": o.get("title", ""),
                "price": price,
                "area": area,
                "floor": o.get("floor", 0),
                "district": o.get("districtName", ""),
                "url": f"https://krisha.kz/a/show/{o.get('id', '')}",
                "price_m2": round(price / area),
                "source": "krisha_api",
            })
        return result
    except Exception:
        return []


def _score_to_price_adj(score: int) -> float:
    """Детерминированная корректировка цены по condition_score."""
    if score >= 9:
        return 10.0
    if score >= 7:
        return 3.0
    if score >= 5:
        return 0.0
    if score >= 3:
        return -8.0
    return -15.0


# ---------------------------------------------------------------------------
# NODE 3: analyze_photos
# GPT-4o Vision анализирует все фото из объявления
# ---------------------------------------------------------------------------
def analyze_photos(state: AgentState) -> AgentState:
    data = state.get("listing_data")
    if not data or not data.get("photo_urls"):
        return {**state, "photo_analysis": None}

    # Скачиваем все доступные фото
    image_contents = []
    for photo_url in data["photo_urls"]:
        try:
            r = requests.get(photo_url, timeout=10)
            if r.status_code == 200:
                img_b64 = base64.b64encode(r.content).decode("utf-8")
                mime = "image/jpeg" if photo_url.endswith(".jpg") else "image/webp"
                image_contents.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img_b64}"}
                })
        except Exception:
            continue

    if not image_contents:
        return {**state, "photo_analysis": None}

    prompt = f"""Ты строгий эксперт по оценке недвижимости в Алматы. Смотришь на {len(image_contents)} фото из объявления о продаже квартиры.

Оцени состояние по шкале 1-10. Будь строгим — большинство квартир на рынке получают 4-6, только реально хорошие 7+.

Шкала оценки:
1-2: аварийное состояние, жить невозможно без капитального ремонта
3-4: советский фонд, подъезд убитый, обои/краска отваливаются, старая сантехника, требует полного ремонта
5-6: жилое состояние, косметический ремонт, ничего не обновлялось последние 10+ лет
7-8: свежий ремонт последних лет, современные материалы, аккуратно
9-10: евроремонт или новостройка с дизайнерской отделкой

Учитывай:
- Состояние внутри квартиры (стены, полы, потолки, двери, окна, сантехника)
- Если есть фото подъезда или лестницы — обязательно учти, это сильно влияет на стоимость
- Если подъезд убитый, а внутри косметика — не ставь выше 4
- НЕ путай качество съёмки с состоянием квартиры — тёмное или размытое фото не означает плохой ремонт, оценивай только то что видно на поверхностях

Верни JSON с полями:
- condition_score: итоговая оценка 1-10
- issues: список видимых проблем (если нет — пустой список)
- low_confidence: true если фото настолько тёмные/размытые что невозможно оценить состояние, или это рендер новостройки, иначе false
- summary: 2-3 предложения о состоянии квартиры и дома

Верни ТОЛЬКО JSON."""

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}] + image_contents}]

    try:
        response = client.chat.completions.create(
            model=config.MODEL_SMART,
            messages=messages,
            temperature=config.TEMP_ANALYZE,
            max_tokens=config.MAX_TOKENS_ANALYZE,
        )
        content = response.choices[0].message.content
        if not content:
            return {**state, "photo_analysis": None}

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            return {**state, "photo_analysis": None}

        analysis = json.loads(json_match.group())

        # Корректировка цены считается детерминированно, не моделью
        score = analysis.get("condition_score", 5)
        low_confidence = analysis.get("low_confidence", False)
        analysis["price_adjustment_pct"] = 0.0 if low_confidence else _score_to_price_adj(score)

        # Нужен ли ремонт
        if score >= 7:
            analysis["renovation_needed"] = "не нужен"
        elif score >= 5:
            analysis["renovation_needed"] = "косметический"
        else:
            analysis["renovation_needed"] = "полный"

        return {**state, "photo_analysis": analysis}
    except Exception:
        return {**state, "photo_analysis": None}


# Маппинг названий районов на URL-алиасы Krisha
DISTRICT_ALIASES = {
    "Алатауский":    "almaty-alatauskij",
    "Алмалинский":   "almaty-almalinskij",
    "Ауэзовский":    "almaty-aujezovskij",
    "Бостандыкский": "almaty-bostandykskij",
    "Жетысуский":    "almaty-zhetysuskij",
    "Медеуский":     "almaty-medeuskij",
    "Наурызбайский": "almaty-nauryzbajskiy",
    "Турксибский":   "almaty-turksibskij",
}


# ---------------------------------------------------------------------------
# Вспомогательная: реальная цена м² по конкретному району из живых объявлений
# ---------------------------------------------------------------------------
def _fetch_market_price_m2(rooms: int = 0, district: str = "", area: float = 0) -> tuple[float, float, float, int, str]:
    """
    Возвращает (avg_m2, median_m2, trend_pct, sample_size, source).

    Приоритет источников:
    1. SQLite база (если есть ≥50 объявлений) — самые точные данные
    2. Live скрапинг первой страницы Krisha по району (~22 объявления)
    3. Krisha Analytics API по всему Алматы (fallback)
    """
    # --- Источник 1: локальная SQLite база ---
    if get_total_count() >= 50:
        stats = get_price_stats(district, rooms, area)
        if stats and stats.get("sample_size", 0) >= 10:
            return (
                stats["avg_price_m2"],
                stats["median_price_m2"],
                0.0,
                stats["sample_size"],
                f"локальная база Krisha ({stats['sample_size']} объявл., медиана без выбросов)",
            )

    # --- Источник 2: live скрапинг по району ---
    alias = DISTRICT_ALIASES.get(district, "")
    if alias:
        prices_m2 = _scrape_district_prices(alias, rooms)
        if len(prices_m2) >= 5:
            avg = sum(prices_m2) / len(prices_m2)
            median = sorted(prices_m2)[len(prices_m2) // 2]
            return avg, median, 0.0, len(prices_m2), f"live Krisha.kz ({len(prices_m2)} объявл., 1 страница)"

    # --- Источник 3: Krisha Analytics API ---
    try:
        resp = requests.get(
            config.KRISHA_ANALYTICS_URL,
            params={"id": 0, "rooms": rooms if rooms else "", "buildingType": "", "mode": "long", "geo": 2},
            headers=HEADERS, timeout=10,
        )
        raw = resp.json()
        series = raw if isinstance(raw, list) else []
        almaty = [p for p in series if p.get("geo") == 2]
        series = almaty if almaty else series
        recent = series[-24:] if len(series) >= 24 else series
        prices = [p["average_kzt"] for p in recent if p.get("average_kzt")]
        if not prices:
            raise ValueError()
        avg = sum(prices) / len(prices)
        median = sorted(prices)[len(prices) // 2]
        trend = 0.0
        if len(prices) >= 8:
            r_avg = sum(prices[-4:]) / 4
            p_avg = sum(prices[-8:-4]) / 4
            trend = round((r_avg - p_avg) / p_avg * 100, 2) if p_avg else 0.0
        return avg, median, trend, len(prices), "Krisha Analytics API (весь Алматы, 6 мес.)"
    except Exception:
        return 800_000.0, 800_000.0, 0.0, 0, "fallback (нет данных)"


def _scrape_district_prices(alias: str, rooms: int = 0) -> list[float]:
    """Скрапит текущие объявления по URL-алиасу района и возвращает список цен м²."""
    url = f"https://krisha.kz/prodazha/kvartiry/{alias}/"
    params = {}
    if rooms:
        params["das[live.rooms]"] = rooms

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception:
        return []

    prices_m2 = []
    for card in soup.find_all(class_="a-card__descr"):
        title_el = card.find(class_="a-card__title")
        price_el = card.find(class_="a-card__price")
        if not title_el or not price_el:
            continue

        # Площадь из заголовка: "2-комнатная квартира · 56.2 м² · 12/18 этаж"
        area_m = re.search(r"([\d.]+)\s*м²", title_el.text)
        area = float(area_m.group(1)) if area_m else 0

        # Цена: "51 000 000 〒"
        price_digits = re.sub(r"[^\d]", "", price_el.text)
        price = int(price_digits) if 6 <= len(price_digits) <= 12 else 0

        if area > 15 and price > 1_000_000:
            prices_m2.append(price / area)

    return prices_m2


# ---------------------------------------------------------------------------
# NODE 4: evaluate_price
# Сравнивает цену объявления с рыночной
# ---------------------------------------------------------------------------
def evaluate_price(state: AgentState) -> AgentState:
    data = state.get("listing_data")
    similar = state.get("similar_listings", [])
    photo = state.get("photo_analysis")

    if not data:
        return {**state, "price_stats": {}, "price_evaluation": None}

    # Рыночная цена — реальные объявления по конкретному району и похожей площади
    rooms = data.get("rooms", 0)
    district = data.get("district", "")
    area = data.get("area", 0)
    avg_m2, median_m2, trend_pct, sample_size, data_source = _fetch_market_price_m2(rooms, district, area)

    # --- Корректировки к рыночной цене ---
    # 1. Структурные: этаж, год постройки, тип дома
    description = data.get("description", "")
    year, building_type = _parse_building_info(description)
    floor = data.get("floor", 0)
    floors_total = data.get("floors_total", 0)
    struct_adj = _compute_structural_adjustments(floor, floors_total, year, building_type)

    # 2. Фото: состояние ремонта
    photo_adj = photo.get("price_adjustment_pct", 0) if photo else 0

    total_adj_pct = struct_adj["total_structural_adj_pct"] + photo_adj
    fair_m2 = avg_m2 * (1 + total_adj_pct / 100)

    area = data.get("area", 60)
    fair_min = fair_m2 * area * 0.92
    fair_max = fair_m2 * area * 1.08
    listing_price = data.get("price", 0)
    listing_m2 = data.get("price_m2", 0)

    # Вердикт
    if listing_price < fair_min:
        verdict = "выгодно"
        verdict_detail = f"Цена на {round((fair_min - listing_price) / 1_000_000, 1)} млн ниже рынка"
    elif listing_price > fair_max:
        overpriced_pct = round((listing_price - fair_max) / fair_max * 100)
        verdict = "дорого"
        verdict_detail = f"Цена завышена на ~{overpriced_pct}%"
    else:
        verdict = "рыночная цена"
        verdict_detail = "Цена соответствует рынку"

    price_stats = {
        "avg_price_m2": round(avg_m2),
        "median_price_m2": round(median_m2),
        "fair_price_m2": round(fair_m2),
        "trend_4w_pct": trend_pct,
        "listings_count": len(similar),
        "sample_size": sample_size,
        "source": data_source,
    }

    price_evaluation = {
        "fair_min": round(fair_min),
        "fair_max": round(fair_max),
        "listing_price": listing_price,
        "listing_price_m2": listing_m2,
        "market_price_m2": round(avg_m2),
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "photo_adjustment_pct": photo_adj,
        "structural_adjustments": struct_adj,
        "total_adjustment_pct": round(total_adj_pct, 1),
    }

    market_comparison = _compare_with_similar(listing_m2, data.get("price", 0), similar)

    return {**state, "price_stats": price_stats, "price_evaluation": price_evaluation, "market_comparison": market_comparison}


def _compare_with_similar(listing_m2: int, listing_price: int, similar: list) -> dict:
    """Сравнивает текущее объявление с похожими по цене м²."""
    valid = [s for s in similar if s.get("price_m2", 0) > 0]
    if not valid:
        return {}

    prices_m2 = sorted([s["price_m2"] for s in valid])
    cheaper = [s for s in valid if s["price_m2"] < listing_m2]
    position = len(cheaper) + 1  # место по цене м² (1 = самое дешёвое)
    total = len(valid) + 1       # +1 включая текущее

    if position == 1:
        verdict = "лучшая сделка"
        detail = f"Самая низкая цена м² среди {total} похожих объявлений"
        best_alt = None
    elif position <= max(2, total // 2):
        verdict = "одна из лучших"
        detail = f"{position}-е место из {total} по цене м²"
        best_alt = None
    else:
        # Найдём лучшую альтернативу
        best = min(valid, key=lambda x: x["price_m2"])
        diff_pct = round((listing_m2 - best["price_m2"]) / best["price_m2"] * 100)
        verdict = "есть варианты лучше"
        detail = f"{position}-е место из {total} по цене м², есть на {diff_pct}% дешевле"
        best_alt = best

    return {
        "verdict": verdict,
        "detail": detail,
        "position": position,
        "total": total,
        "best_alternative": best_alt,
    }


# ---------------------------------------------------------------------------
# NODE 5: generate_recommendation
# GPT-4o объясняет вердикт с конкретными цифрами и аргументами
# ---------------------------------------------------------------------------
def generate_recommendation(state: AgentState) -> AgentState:
    data = state.get("listing_data", {})
    similar = state.get("similar_listings", [])
    price_stats = state.get("price_stats", {})
    price_eval = state.get("price_evaluation", {})
    photo = state.get("photo_analysis", {})

    top_similar = similar[:3]
    similar_text = "\n".join(
        f"- {l.get('title','')}: {l['price']:,}₸, {l['area']}м², {l.get('price_m2',0):,}₸/м²"
        for l in top_similar
    ) or "нет данных"

    photo_text = ""
    if photo:
        photo_text = f"""
Состояние квартиры по фото:
- Ремонт: {photo.get('condition_score')}/10
- Планировка: {photo.get('layout')}
- Освещённость: {photo.get('lighting')}
- Проблемы: {', '.join(photo.get('issues', [])) or 'не обнаружены'}
- Корректировка цены: {photo.get('price_adjustment_pct', 0):+.0f}%
- {photo.get('summary', '')}
"""

    struct_adj = price_eval.get("structural_adjustments", {})
    struct_lines = []
    if struct_adj.get("floor_adj_pct"):
        struct_lines.append(f"  • Этаж {data.get('floor')}/{data.get('floors_total')}: {struct_adj['floor_adj_pct']:+.0f}%")
    if struct_adj.get("year") and struct_adj.get("year_adj_pct") != 0:
        struct_lines.append(f"  • Год постройки {struct_adj['year']}: {struct_adj['year_adj_pct']:+.0f}%")
    if struct_adj.get("building_type") and struct_adj.get("building_type_adj_pct") != 0:
        struct_lines.append(f"  • Тип дома ({struct_adj['building_type']}): {struct_adj['building_type_adj_pct']:+.0f}%")
    if struct_adj.get("total_structural_adj_pct"):
        struct_lines.append(f"  • Итого структурная корректировка: {struct_adj['total_structural_adj_pct']:+.1f}%")
    struct_text = "\n".join(struct_lines) if struct_lines else "  нет данных"

    prompt = f"""Ты эксперт по рынку недвижимости Алматы. Объясни покупателю — стоит ли брать эту квартиру.

ОБЪЯВЛЕНИЕ:
- {data.get('title', '')}
- Цена: {data.get('price', 0):,}₸ ({data.get('price_m2', 0):,}₸/м²)
- Площадь: {data.get('area', 0)}м², {data.get('floor', 0)}/{data.get('floors_total', 0)} этаж
- Район: {data.get('district', '')}
- URL: {data.get('url', '')}

РЫНОК:
- Средняя цена м²: {price_stats.get('avg_price_m2', 0):,}₸
- Справедливая цена объекта (после корректировок): {price_eval.get('fair_min', 0):,}₸ — {price_eval.get('fair_max', 0):,}₸
- Вердикт: {price_eval.get('verdict', '')} — {price_eval.get('verdict_detail', '')}

КОРРЕКТИРОВКИ К РЫНОЧНОЙ ЦЕНЕ:
{struct_text}
  • Фото/ремонт: {price_eval.get('photo_adjustment_pct', 0):+.0f}%
  • Итого все корректировки: {price_eval.get('total_adjustment_pct', 0):+.1f}%

ПОХОЖИЕ ОБЪЯВЛЕНИЯ НА РЫНКЕ:
{similar_text}
{photo_text}

Напиши подробную рекомендацию в формате:

**Вердикт:** [одно предложение]

**Почему такая оценка:**
[Объясни цифрами — цена м² объявления vs рынок, как это соотносится с похожими объявлениями]

**Что влияет на цену:**
[Факторы: этаж, состояние ремонта из фото, район, площадь — как каждый влияет]

**Рекомендация по торгу:**
[Конкретно — торговаться ли, на сколько, аргументы для продавца]

**Итог:**
[Брать / подождать / искать дальше — с объяснением]

Пиши по-русски, конкретно, с цифрами. Без воды."""

    response = client.chat.completions.create(
        model=config.MODEL_SMART,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.TEMP_RECOMMEND,
        max_tokens=config.MAX_TOKENS_RECOMMEND,
        top_p=config.TOP_P,
    )

    return {**state, "recommendation": response.choices[0].message.content, "awaiting_feedback": True}


# ---------------------------------------------------------------------------
# NODE 6: human_review
# Обрабатывает уточняющий вопрос после рекомендации
# ---------------------------------------------------------------------------
def human_review(state: AgentState) -> AgentState:
    feedback = (state.get("user_feedback") or "").strip()
    if not feedback:
        return {**state, "awaiting_feedback": True}

    # Отвечаем на уточняющий вопрос через LLM
    context = f"""
Объявление: {state.get('listing_data', {}).get('title', '')}
Цена: {state.get('listing_data', {}).get('price', 0):,}₸
Вердикт: {state.get('price_evaluation', {}).get('verdict', '')}
Рекомендация агента: {state.get('recommendation', '')}
"""
    response = client.chat.completions.create(
        model=config.MODEL_TEXT,
        messages=[
            {"role": "system", "content": f"Ты эксперт по недвижимости Алматы. Контекст анализа:\n{context}"},
            {"role": "user", "content": feedback},
        ],
        temperature=0.3,
        max_tokens=500,
    )
    new_recommendation = state.get("recommendation", "") + f"\n\n---\n**Ответ на ваш вопрос:** {response.choices[0].message.content}"
    return {**state, "recommendation": new_recommendation, "user_feedback": None, "awaiting_feedback": True}
