"""
Bulk scraper объявлений Krisha.kz по всем районам Алматы.
Запуск: python data/scraper.py
Собирает ~5,000-6,000 объявлений и сохраняет в SQLite.
"""
import sys
import os
import re
import time
import random
import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data.database import init_db, upsert_listing, get_total_count, get_district_summary

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "scraper_checkpoint.json")


def load_checkpoint() -> set:
    """Загружает список уже обработанных комбинаций (district, rooms, sort_by)."""
    if not os.path.exists(CHECKPOINT_PATH):
        return set()
    try:
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
        return {tuple(item) for item in data.get("completed", [])}
    except Exception:
        return set()


def _checkpoint_key(district: str, rooms, sort_by: str) -> tuple:
    return (district, str(rooms), sort_by)


def save_checkpoint(completed: set):
    """Сохраняет прогресс на диск."""
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"completed": [list(item) for item in completed]}, f)


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "ru-RU,ru;q=0.9",
    "Accept": "text/html,application/xhtml+xml",
}

DISTRICTS = {
    "Алатауский":    "almaty-alatauskij",
    "Алмалинский":   "almaty-almalinskij",
    "Ауэзовский":    "almaty-aujezovskij",
    "Бостандыкский": "almaty-bostandykskij",
    "Жетысуский":    "almaty-zhetysuskij",
    "Медеуский":     "almaty-medeuskij",
    "Наурызбайский": "almaty-nauryzbajskiy",
    "Турксибский":   "almaty-turksibskij",
}

ROOMS = [1, 2, 3, 4, 5, None]  # None = без фильтра (студии, апартаменты)
MAX_PAGES = 200  # достаточно для покрытия всех объявлений в районе

# Одна сортировка по цене — стабильная пагинация без дублей
SORT_ORDERS = ["price-asc"]


def scrape_page(district_name: str, alias: str, rooms, page: int, sort_by: str = "date") -> list[dict]:
    """Скрапит одну страницу поиска и возвращает список объявлений."""
    url = f"https://krisha.kz/prodazha/kvartiry/{alias}/"
    params = {"page": page, "sort_by": sort_by}
    if rooms is not None:
        params["das[live.rooms]"] = rooms

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "lxml")
    except Exception:
        return []

    listings = []
    for card in soup.find_all(class_="a-card__descr"):
        try:
            title_el = card.find(class_="a-card__title")
            price_el = card.find(class_="a-card__price")
            if not title_el or not price_el:
                continue

            # ID из href: /a/show/123456789
            href = title_el.get("href", "")
            id_m = re.search(r"/a/show/(\d+)", href)
            listing_id = id_m.group(1) if id_m else ""
            if not listing_id:
                continue

            title = title_el.text.strip()

            # Площадь
            area_m = re.search(r"([\d.]+)\s*м²", title)
            area = float(area_m.group(1)) if area_m else 0

            # Этаж
            floor_m = re.search(r"(\d+)/(\d+)\s*этаж", title)
            floor = int(floor_m.group(1)) if floor_m else 0
            floors_total = int(floor_m.group(2)) if floor_m else 0

            # Цена
            price_digits = re.sub(r"[^\d]", "", price_el.text)
            price = int(price_digits) if 6 <= len(price_digits) <= 12 else 0

            if area < 15 or price < 1_000_000:
                continue

            price_m2 = price / area

            # Фильтр явных выбросов (< 200к или > 5млн ₸/м²)
            if not (200_000 < price_m2 < 5_000_000):
                continue

            listings.append({
                "listing_id":   listing_id,
                "district":     district_name,
                "rooms":        rooms,
                "area":         area,
                "floor":        floor,
                "floors_total": floors_total,
                "price":        price,
                "price_m2":     round(price_m2, 2),
            })
        except Exception:
            continue

    return listings


def has_next_page(soup: BeautifulSoup) -> bool:
    """Проверяет есть ли следующая страница пагинации."""
    pagination = soup.find(class_=re.compile("pagination"))
    if not pagination:
        return False
    next_btn = pagination.find("a", class_=re.compile("next|»"))
    return next_btn is not None


def run_scraper(max_pages: int = MAX_PAGES):
    init_db()

    completed = load_checkpoint()
    total_tasks = len(DISTRICTS) * len(ROOMS) * len(SORT_ORDERS)
    skipped = len(completed)

    if skipped:
        print(f"Найден checkpoint — пропускаю {skipped} уже обработанных комбинаций.")
    print(f"Начинаю сбор данных: {len(DISTRICTS)} районов × {len(ROOMS)} типов комнат × {len(SORT_ORDERS)} сортировки")
    print(f"Максимум {max_pages} страниц на комбинацию\n")

    saved = 0
    rooms_label = {None: "все"}
    pbar = tqdm(total=total_tasks, initial=skipped, desc="Прогресс", unit="комбо")

    # Для прохода без фильтра: знаем уже существующие listing_id чтобы не дублировать
    existing_ids: set | None = None

    for district_name, alias in DISTRICTS.items():
        for rooms in ROOMS:
            for sort_by in SORT_ORDERS:
                key = (district_name, str(rooms), sort_by)

                if key in completed:
                    continue

                # Для прохода без фильтра грузим существующие ID один раз
                if rooms is None and existing_ids is None:
                    from data.database import get_conn
                    conn = get_conn()
                    rows = conn.execute("SELECT listing_id FROM listings").fetchall()
                    existing_ids = {r["listing_id"] for r in rows}
                    conn.close()

                combo_count = 0

                for page in range(1, max_pages + 1):
                    listings = scrape_page(district_name, alias, rooms, page, sort_by)

                    if not listings:
                        break

                    for lst in listings:
                        # Без фильтра комнат — сохраняем только новые
                        if rooms is None and lst["listing_id"] in existing_ids:
                            continue
                        upsert_listing(lst)
                        if existing_ids is not None:
                            existing_ids.add(lst["listing_id"])
                        combo_count += 1
                        saved += 1

                    time.sleep(random.uniform(0.8, 1.5))

                    if len(listings) < 10:
                        break

                completed.add(key)
                save_checkpoint(completed)

                label = rooms_label.get(rooms, f"{rooms}к")
                pbar.set_postfix({"всего": saved, "последний": f"{district_name} {label}"})
                pbar.update(1)

                time.sleep(random.uniform(0.5, 1.0))

    pbar.close()
    clear_checkpoint()

    total = get_total_count()
    print(f"\n✓ Готово! Сохранено {total:,} объявлений в базе.\n")

    print("Сводка по районам:")
    print(f"{'Район':<20} {'Комн':>5} {'Объявл':>8} {'Ср. цена м²':>12}")
    print("-" * 50)
    for row in get_district_summary():
        rooms_label = str(row['rooms']) if row['rooms'] else "все"
        print(f"{row['district']:<20} {rooms_label:>5} {row['cnt']:>8} {row['avg_m2']:>12,.0f}₸")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=MAX_PAGES, help="Страниц на комбинацию × сортировку (default=5)")
    args = parser.parse_args()
    run_scraper(max_pages=args.pages)
