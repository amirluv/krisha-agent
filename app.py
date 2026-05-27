import streamlit as st
import os

import config
from agent.graph import run_agent


@st.cache_resource(show_spinner=False)
def ensure_chroma():
    """Пересоздаёт ChromaDB если её нет (первый запуск на сервере)."""
    chroma_dir = config.CHROMA_DIR
    needs_index = (
        not os.path.isdir(chroma_dir)
        or not any(os.scandir(chroma_dir))
    )
    if needs_index:
        from rag.indexer import index_from_sqlite
        with st.spinner("Первый запуск: индексируем базу объявлений (~3 мин)..."):
            index_from_sqlite()


ensure_chroma()

st.set_page_config(
    page_title="KrishaAgent — AI анализ квартир",
    page_icon="🏢",
    layout="wide",
)

# --- Sidebar ---
with st.sidebar:
    st.markdown("**Как это работает:**")
    st.markdown("""
1. Вставляешь ссылку объявления
2. Агент скачивает данные и фото
3. Сравнивает с рынком
4. Объясняет — почему цена справедлива или нет
""")

st.markdown("""
<style>
.verdict-good { background:#dcfce7; border-left:4px solid #16a34a; padding:14px 18px; border-radius:8px; font-size:18px; font-weight:700; color:#14532d; }
.verdict-bad  { background:#fee2e2; border-left:4px solid #dc2626; padding:14px 18px; border-radius:8px; font-size:18px; font-weight:700; color:#7f1d1d; }
.verdict-mid  { background:#fef9c3; border-left:4px solid #ca8a04; padding:14px 18px; border-radius:8px; font-size:18px; font-weight:700; color:#713f12; }
.stat-card    { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:16px; text-align:center; }
.stat-value   { font-size:22px; font-weight:700; color:#1e293b; }
.stat-label   { font-size:12px; color:#64748b; margin-top:4px; }
.listing-box  { background:#f1f5f9; border-radius:8px; padding:14px 18px; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

# --- Заголовок ---
st.title("🏢 KrishaAgent")
st.caption("Вставь ссылку с Krisha.kz — агент скажет, стоит ли брать и почему")

# --- Инициализация ---
if "agent_state" not in st.session_state:
    st.session_state.agent_state = None

# --- Ввод URL ---
url_input = st.text_input(
    "Ссылка на объявление",
    placeholder="https://krisha.kz/a/show/12345678",
    label_visibility="collapsed",
)

col1, col2 = st.columns([1, 5])
with col1:
    analyze_btn = st.button("Анализировать", type="primary", use_container_width=True)
with col2:
    if st.session_state.agent_state:
        if st.button("Новый поиск"):
            st.session_state.agent_state = None
            st.rerun()

# --- Запуск ---
if analyze_btn and url_input.strip():
    with st.spinner("Агент загружает объявление и анализирует рынок..."):
        try:
            state = run_agent(url_input.strip())
            st.session_state.agent_state = state
        except Exception as e:
            st.error(f"Ошибка: {e}")
            st.stop()

# --- Результаты ---
state = st.session_state.agent_state
if not state:
    st.stop()

if state.get("error"):
    st.error(state["error"])
    st.stop()

data = state.get("listing_data", {})
price_eval = state.get("price_evaluation", {})
price_stats = state.get("price_stats", {})
photo = state.get("photo_analysis")
market_cmp = state.get("market_comparison", {})
verdict = price_eval.get("verdict", "")

st.divider()

# --- Заголовок объявления ---
if data.get("title"):
    st.subheader(data["title"])
    st.caption(f"[Открыть на Krisha.kz]({data.get('url', '')})")

# --- Вердикт ---
verdict_class = {"выгодно": "verdict-good", "дорого": "verdict-bad"}.get(verdict, "verdict-mid")
verdict_icon = {"выгодно": "✅", "дорого": "⚠️"}.get(verdict, "📊")
verdict_detail = price_eval.get("verdict_detail", "")

st.markdown(
    f'<div class="{verdict_class}">{verdict_icon} {verdict.upper()} — {verdict_detail}</div>',
    unsafe_allow_html=True,
)

# --- Сравнение с похожими ---
if market_cmp:
    cmp_icons = {"лучшая сделка": "🏆", "одна из лучших": "👍", "есть варианты лучше": "💡"}
    cmp_icon = cmp_icons.get(market_cmp["verdict"], "📊")
    cmp_detail = market_cmp["detail"]
    best_alt = market_cmp.get("best_alternative")
    alt_text = ""
    if best_alt:
        pm = best_alt.get("price", 0) // 1_000_000
        m2 = int(best_alt.get("price_m2", 0))
        area = best_alt.get("area", 0)
        url = best_alt.get("url", "")
        if url:
            alt_text = f' → [посмотреть альтернативу]({url}) ({pm} млн · {m2:,}₸/м²)'
        else:
            alt_text = f' → альтернатива: {pm} млн · {m2:,}₸/м² · {area}м²'
    st.caption(f"{cmp_icon} **{market_cmp['verdict'].upper()}** — {cmp_detail}{alt_text}")

st.markdown("")

# --- Метрики ---
c1, c2, c3, c4 = st.columns(4)
listing_price = data.get("price", 0)
listing_m2 = data.get("price_m2", 0)
market_m2 = price_stats.get("avg_price_m2", 0)
fair_min = price_eval.get("fair_min", 0)
fair_max = price_eval.get("fair_max", 0)

diff_pct = round((listing_m2 - market_m2) / market_m2 * 100) if market_m2 else 0
diff_str = f"{diff_pct:+}% от рынка"

with c1:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{listing_price // 1_000_000} млн ₸</div><div class="stat-label">Цена продавца</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{listing_m2:,} ₸/м²</div><div class="stat-label">Цена м² объявления</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{market_m2:,} ₸/м²</div><div class="stat-label">Средний рынок</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="stat-card"><div class="stat-value">{fair_min // 1_000_000}–{fair_max // 1_000_000} млн</div><div class="stat-label">Справедливая цена</div></div>', unsafe_allow_html=True)

st.markdown("")

# --- Основной контент ---
left, right = st.columns([3, 2])

with left:
    st.subheader("Анализ агента")
    rec = state.get("recommendation", "")
    if rec:
        st.markdown(rec)

with right:
    # Анализ фото
    if photo:
        st.subheader("Состояние квартиры")
        score = photo.get("condition_score", 0)
        renovation = photo.get("renovation_needed", "—")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Ремонт", f"{score}/10")
        with col_b:
            st.metric("Ремонт нужен", renovation)
        issues = photo.get("issues", [])
        if issues:
            st.write("**Обнаружено:**")
            for issue in issues:
                st.write(f"- {issue}")
        if photo.get("summary"):
            st.caption(photo["summary"])
    else:
        st.info("Фото не найдены в объявлении")

    # Похожие объявления
    similar = state.get("similar_listings", [])
    if similar:
        st.subheader(f"Похожие объявления ({len(similar)})")
        for lst in similar[:4]:
            pm = lst.get("price", 0) // 1_000_000
            m2 = lst.get("price_m2", 0)
            area = lst.get("area", 0)
            url = lst.get("url", "")
            title = lst.get("title") or f"{area}м²"
            line = f"{pm} млн · {m2:,}₸/м² · {area}м²"
            if url:
                st.markdown(f"[{title}]({url})  \n`{line}`")
            else:
                st.markdown(f"**{title}**  \n`{line}`")

# --- Фото из объявления ---
photo_urls = data.get("photo_urls", [])
if photo_urls:
    with st.expander(f"Фото из объявления ({len(photo_urls)} шт.)", expanded=False):
        for row_start in range(0, len(photo_urls), 3):
            row_urls = photo_urls[row_start:row_start + 3]
            cols = st.columns(3)
            for i, url in enumerate(row_urls):
                with cols[i]:
                    st.image(url, use_container_width=True)

