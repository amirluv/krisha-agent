"""
SQLite база данных для хранения объявлений Krisha.kz.
Используется для расчёта рыночных цен по районам.
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "krisha_listings.db")


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS listings (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                listing_id  TEXT UNIQUE,
                district    TEXT,
                rooms       INTEGER,
                area        REAL,
                floor       INTEGER,
                floors_total INTEGER,
                price       INTEGER,
                price_m2    REAL,
                scraped_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_district_rooms ON listings(district, rooms)")
        conn.commit()


def upsert_listing(listing: dict):
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO listings (listing_id, district, rooms, area, floor, floors_total, price, price_m2)
            VALUES (:listing_id, :district, :rooms, :area, :floor, :floors_total, :price, :price_m2)
            ON CONFLICT(listing_id) DO UPDATE SET
                price    = excluded.price,
                price_m2 = excluded.price_m2,
                scraped_at = datetime('now')
        """, listing)
        conn.commit()


def get_price_stats(district: str, rooms: int, area: float = 0) -> dict:
    """
    Возвращает статистику цен м² для района и типа квартиры.
    Если передана площадь — фильтрует похожие объявления (±30%).
    Фильтрует выбросы методом IQR. Использует медиану как основной показатель.
    """
    area_filter = ""
    area_params: tuple = ()
    if area > 0:
        area_min, area_max = area * 0.7, area * 1.3
        area_filter = "AND area BETWEEN ? AND ?"
        area_params = (area_min, area_max)

    with get_conn() as conn:
        rows = conn.execute(f"""
            SELECT price_m2 FROM listings
            WHERE district = ? AND rooms = ? AND price_m2 > 0 {area_filter}
            ORDER BY price_m2
        """, (district, rooms) + area_params).fetchall()

    if not rows:
        with get_conn() as conn:
            rows = conn.execute(f"""
                SELECT price_m2 FROM listings
                WHERE district = ? AND price_m2 > 0 {area_filter}
                ORDER BY price_m2
            """, (district,) + area_params).fetchall()

    if not rows:
        return {}

    prices = [r["price_m2"] for r in rows]
    n = len(prices)

    # IQR фильтрация выбросов
    q1 = prices[n // 4]
    q3 = prices[3 * n // 4]
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    trimmed = [p for p in prices if lower <= p <= upper]

    if not trimmed:
        trimmed = prices  # если IQR убрал всё — берём исходные

    avg    = sum(trimmed) / len(trimmed)
    median = trimmed[len(trimmed) // 2]
    p25    = trimmed[len(trimmed) // 4]
    p75    = trimmed[3 * len(trimmed) // 4]
    removed = n - len(trimmed)

    return {
        "avg_price_m2":    round(avg),
        "median_price_m2": round(median),
        "p25_price_m2":    round(p25),
        "p75_price_m2":    round(p75),
        "min_price_m2":    round(trimmed[0]),
        "max_price_m2":    round(trimmed[-1]),
        "sample_size":     n,
        "outliers_removed": removed,
        "district":        district,
        "rooms":           rooms,
    }


def get_total_count() -> int:
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM listings").fetchone()[0]


def get_district_summary() -> list[dict]:
    """Сводка по районам — сколько объявлений и средняя цена."""
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT district, rooms, COUNT(*) as cnt,
                   ROUND(AVG(price_m2)) as avg_m2
            FROM listings
            WHERE price_m2 > 0
            GROUP BY district, rooms
            ORDER BY district, rooms
        """).fetchall()
    return [dict(r) for r in rows]
