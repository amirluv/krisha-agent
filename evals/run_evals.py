"""
Автоматизированный прогон эвалюаций KrishaAgent.

Запуск:
    python evals/run_evals.py --dataset evals/golden_dataset.json --output evals/results.json
    python evals/run_evals.py --dataset evals/golden_dataset.json --output evals/results_mini.json --config A

Конфиги:
    A (baseline) — gpt-4o-mini для recommendation, gpt-4o для vision
    B (production) — gpt-4o для recommendation, gpt-4o для vision (текущий)

Метрики:
    1. relevance_score   — покрывает ли рекомендация ключевые факторы из expected
    2. llm_judge_score   — GPT-4o-mini оценивает качество рекомендации по шкале 1-5
"""

import sys
import os
import json
import argparse
import time
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config
from openai import OpenAI

client = OpenAI(api_key=config.OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Генерация рекомендации из pre-loaded данных (без скрапинга и Vision)
# ---------------------------------------------------------------------------

def generate_recommendation_from_data(
    listing_data: dict,
    price_stats: dict,
    photo_analysis: dict,
    model: str,
) -> dict:
    """Генерирует оценку цены и рекомендацию из готовых данных."""
    area = listing_data.get("area", 60)
    listing_m2 = listing_data.get("price_m2", 0)
    avg_m2 = price_stats.get("avg_price_m2", 800000)
    listing_price = listing_data.get("price", 0)

    # Структурные корректировки
    description = listing_data.get("description", "")
    year = _parse_year(description)
    building_type = _parse_building_type(description)
    floor = listing_data.get("floor", 0)
    floors_total = listing_data.get("floors_total", 0)

    floor_adj = _floor_adjustment(floor, floors_total)
    year_adj = _year_adjustment(year)
    bt_adj = _building_type_adjustment(building_type)
    photo_adj = photo_analysis.get("price_adjustment_pct", 0)
    total_adj = floor_adj + year_adj + bt_adj + photo_adj

    fair_m2 = avg_m2 * (1 + total_adj / 100)
    fair_min = round(fair_m2 * area * 0.92)
    fair_max = round(fair_m2 * area * 1.08)

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

    price_evaluation = {
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "fair_min": fair_min,
        "fair_max": fair_max,
        "total_adjustment_pct": round(total_adj, 1),
        "photo_adjustment_pct": photo_adj,
    }

    # Генерация рекомендации через LLM
    prompt = f"""Ты эксперт по рынку недвижимости Алматы. Объясни покупателю — стоит ли брать эту квартиру.

ОБЪЯВЛЕНИЕ:
- {listing_data.get('title', '')}
- Цена: {listing_data.get('price', 0):,}₸ ({listing_m2:,}₸/м²)
- Площадь: {area}м², {floor}/{floors_total} этаж
- Район: {listing_data.get('district', '')}

РЫНОК:
- Средняя цена м²: {avg_m2:,}₸
- Справедливая цена (после корректировок): {fair_min:,}₸ — {fair_max:,}₸
- Вердикт: {verdict} — {verdict_detail}

КОРРЕКТИРОВКИ:
- Этаж: {floor_adj:+.0f}%
- Год постройки ({year}): {year_adj:+.0f}%
- Тип дома ({building_type or 'неизвестно'}): {bt_adj:+.0f}%
- Состояние по фото (оценка {photo_analysis.get('condition_score', '—')}/10): {photo_adj:+.0f}%
- Итого: {total_adj:+.1f}%

СОСТОЯНИЕ КВАРТИРЫ:
- Ремонт нужен: {photo_analysis.get('renovation_needed', '—')}
- Проблемы: {', '.join(photo_analysis.get('issues', [])) or 'не выявлены'}
- {photo_analysis.get('summary', '')}

Напиши рекомендацию:
**Вердикт:** [одно предложение]
**Почему такая оценка:** [объясни цифрами]
**Что влияет на цену:** [факторы]
**Рекомендация по торгу:** [конкретно]
**Итог:** [брать / подождать / искать дальше]

Пиши по-русски, конкретно, с цифрами."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=600,
    )
    recommendation = response.choices[0].message.content

    return {
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "price_evaluation": price_evaluation,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Метрика 1: Relevance Score (детерминированная)
# ---------------------------------------------------------------------------

def compute_relevance_score(recommendation: str, expected: dict) -> float:
    """
    Проверяет покрывает ли рекомендация ключевые факторы.
    Возвращает оценку 0.0–1.0.
    """
    key_factors = expected.get("key_factors", [])
    if not key_factors:
        return 1.0

    rec_lower = recommendation.lower()
    covered = 0

    for factor in key_factors:
        # Извлекаем ключевые слова из описания фактора
        keywords = _extract_keywords(factor)
        if any(kw in rec_lower for kw in keywords):
            covered += 1

    # Также проверяем что вердикт упоминается
    expected_verdict = expected.get("verdict", "")
    verdict_synonyms = {
        "выгодно": ["выгодно", "выгодная", "ниже рынка", "дёшево", "берите"],
        "дорого": ["дорого", "завышена", "переплата", "дороже рынка"],
        "рыночная цена": ["рыночная", "соответствует", "справедливая", "норма"],
    }
    synonyms = verdict_synonyms.get(expected_verdict, [])
    if any(s in rec_lower for s in synonyms):
        covered += 1
        key_factors = key_factors + ["verdict_mention"]  # учитываем в знаменателе

    total = len(key_factors)
    return round(covered / total, 2) if total else 1.0


def _extract_keywords(factor: str) -> list[str]:
    """Выделяет ключевые слова из описания фактора."""
    factor_lower = factor.lower()
    keywords = []

    # Числа и проценты
    import re
    numbers = re.findall(r'\d+', factor_lower)
    keywords.extend(numbers)

    # Ключевые слова
    key_words = [
        "рынок", "ниже", "выше", "год", "этаж", "ремонт", "панел", "монолит",
        "кирпич", "первый", "последний", "новостройк", "завышен", "выгодн",
    ]
    for kw in key_words:
        if kw in factor_lower:
            keywords.append(kw)

    return keywords if keywords else [factor_lower[:10]]


# ---------------------------------------------------------------------------
# Метрика 2: LLM-as-Judge Score
# ---------------------------------------------------------------------------

def compute_llm_judge_score(
    listing_data: dict,
    recommendation: str,
    expected: dict,
    verdict: str,
) -> dict:
    """
    GPT-4o-mini оценивает качество рекомендации по шкале 1-5.
    """
    prompt = f"""Оцени качество рекомендации AI-агента по недвижимости.

ОБЪЯВЛЕНИЕ:
- {listing_data.get('title', '')}
- Цена: {listing_data.get('price', 0):,}₸ ({listing_data.get('price_m2', 0):,}₸/м²)
- Район: {listing_data.get('district', '')}

ОЖИДАЕМЫЙ ВЕРДИКТ: {expected.get('verdict', '')}
ФАКТИЧЕСКИЙ ВЕРДИКТ АГЕНТА: {verdict}

РЕКОМЕНДАЦИЯ АГЕНТА:
{recommendation}

ОЖИДАЕМОЕ ОБОСНОВАНИЕ:
{expected.get('verdict_reasoning', '')}

Оцени рекомендацию по шкале 1-5:
5 — Вердикт верный, объяснение конкретное с цифрами, факторы учтены
4 — Вердикт верный, объяснение хорошее, но не все факторы упомянуты
3 — Вердикт верный, но объяснение поверхностное или без цифр
2 — Вердикт неверный, но объяснение логичное
1 — Вердикт неверный и объяснение неудовлетворительное

Верни JSON: {{"score": <1-5>, "verdict_correct": <true/false>, "reason": "<одно предложение>"}}
Верни ТОЛЬКО JSON."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150,
    )
    content = response.choices[0].message.content.strip()

    import re
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if json_match:
        result = json.loads(json_match.group())
        return result
    return {"score": 1, "verdict_correct": False, "reason": "не удалось разобрать ответ"}


# ---------------------------------------------------------------------------
# Вспомогательные функции (упрощённые из nodes.py)
# ---------------------------------------------------------------------------

def _parse_year(description: str) -> int:
    import re
    m = re.search(r"(?:Год постройки\s*\n+\s*|г\.?\s*п\.?)(\d{4})", description, re.I)
    if not m:
        m = re.search(r"(\d{4})\s*г\.?\s*п", description, re.I)
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2030:
            return y
    return 0


def _parse_building_type(description: str) -> str:
    import re
    m = re.search(r"(монолитный|монолит|кирпичный|кирпич|панельный|панель)", description, re.I)
    if m:
        raw = m.group(1).lower()
        if "монол" in raw:
            return "монолитный"
        elif "кирпич" in raw:
            return "кирпичный"
        elif "панел" in raw:
            return "панельный"
    return ""


def _floor_adjustment(floor: int, floors_total: int) -> float:
    if floor and floors_total:
        if floor == 1:
            return -5.0
        elif floor == floors_total:
            return -3.0
    return 0.0


def _year_adjustment(year: int) -> float:
    if not year:
        return 0.0
    if year >= 2020:
        return 8.0
    elif year >= 2010:
        return 2.0
    elif year >= 2000:
        return 0.0
    elif year >= 1990:
        return -5.0
    elif year >= 1970:
        return -15.0
    elif year >= 1960:
        return -20.0
    elif year >= 1950:
        return -28.0
    return -35.0


def _building_type_adjustment(bt: str) -> float:
    return {"монолитный": 5.0, "кирпичный": 0.0, "панельный": -10.0}.get(bt, 0.0)


# ---------------------------------------------------------------------------
# Основной прогон
# ---------------------------------------------------------------------------

def run_evals(dataset_path: str, output_path: str, config_name: str = "B"):
    with open(dataset_path, encoding="utf-8") as f:
        dataset = json.load(f)

    model = "gpt-4o-mini" if config_name == "A" else "gpt-4o"
    print(f"\nKrishaAgent Evals — Config {config_name} ({model} для рекомендаций)")
    print(f"Датасет: {len(dataset)} примеров\n")

    results = []
    relevance_scores = []
    judge_scores = []
    verdict_correct_count = 0

    for i, example in enumerate(dataset):
        example_id = example["id"]
        listing_data = example["listing_data"]
        price_stats = example["price_stats"]
        photo_analysis = example["photo_analysis"]
        expected = example["expected"]

        print(f"[{i+1:02d}/{len(dataset)}] {example_id} — {example.get('description', '')[:50]}...", end=" ", flush=True)

        try:
            # Генерируем рекомендацию
            output = generate_recommendation_from_data(
                listing_data, price_stats, photo_analysis, model
            )

            # Метрика 1: Relevance
            rel_score = compute_relevance_score(output["recommendation"], expected)

            # Метрика 2: LLM-as-Judge
            judge_result = compute_llm_judge_score(
                listing_data,
                output["recommendation"],
                expected,
                output["verdict"],
            )

            verdict_match = output["verdict"] == expected["verdict"]
            if verdict_match:
                verdict_correct_count += 1

            relevance_scores.append(rel_score)
            judge_scores.append(judge_result["score"])

            result = {
                "id": example_id,
                "description": example.get("description", ""),
                "expected_verdict": expected["verdict"],
                "actual_verdict": output["verdict"],
                "verdict_correct": verdict_match,
                "relevance_score": rel_score,
                "llm_judge_score": judge_result["score"],
                "llm_judge_verdict_correct": judge_result.get("verdict_correct", False),
                "llm_judge_reason": judge_result.get("reason", ""),
                "recommendation_preview": output["recommendation"][:200],
            }
            results.append(result)

            status = "✅" if verdict_match else "❌"
            print(f"{status} вердикт={output['verdict']} | relevance={rel_score:.2f} | judge={judge_result['score']}/5")

        except Exception as e:
            print(f"❌ ОШИБКА: {e}")
            results.append({
                "id": example_id,
                "error": str(e),
                "verdict_correct": False,
                "relevance_score": 0.0,
                "llm_judge_score": 0,
            })

        time.sleep(0.5)  # не спамим API

    # Итоговая статистика
    verdict_accuracy = verdict_correct_count / len(dataset) * 100
    avg_relevance = mean(relevance_scores) if relevance_scores else 0
    avg_judge = mean(judge_scores) if judge_scores else 0

    summary = {
        "config": config_name,
        "model_recommendation": model,
        "total_examples": len(dataset),
        "verdict_accuracy_pct": round(verdict_accuracy, 1),
        "avg_relevance_score": round(avg_relevance, 3),
        "avg_llm_judge_score": round(avg_judge, 2),
        "verdict_correct_count": verdict_correct_count,
    }

    output_data = {"summary": summary, "results": results}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Config {config_name} | Модель рекомендации: {model}")
    print(f"Точность вердиктов:  {verdict_accuracy:.1f}%  ({verdict_correct_count}/{len(dataset)})")
    print(f"Relevance score:     {avg_relevance:.3f}  (0–1)")
    print(f"LLM Judge score:     {avg_judge:.2f}  (1–5)")
    print(f"Результаты: {output_path}")
    print(f"{'='*60}\n")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evals/golden_dataset.json")
    parser.add_argument("--output", default="evals/results.json")
    parser.add_argument(
        "--config",
        choices=["A", "B"],
        default="B",
        help="A = gpt-4o-mini для рекомендации; B = gpt-4o (production)",
    )
    args = parser.parse_args()

    run_evals(args.dataset, args.output, args.config)
