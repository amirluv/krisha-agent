from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import (
    fetch_listing,
    search_similar,
    analyze_photos,
    evaluate_price,
    generate_recommendation,
    human_review,
)


def _listing_loaded(state: AgentState) -> str:
    """Ветвление: объявление загружено успешно?"""
    if state.get("error") or not state.get("listing_data"):
        return "error"
    return "ok"


def _after_human_review(state: AgentState) -> str:
    """После human_review всегда остаёмся в ожидании — конец сессии."""
    return END


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("fetch_listing", fetch_listing)
    graph.add_node("search_similar", search_similar)
    graph.add_node("analyze_photos", analyze_photos)
    graph.add_node("evaluate_price", evaluate_price)
    graph.add_node("generate_recommendation", generate_recommendation)
    graph.add_node("human_review", human_review)

    graph.set_entry_point("fetch_listing")

    # Ветвление: успешно загрузили объявление?
    graph.add_conditional_edges(
        "fetch_listing",
        _listing_loaded,
        {"ok": "search_similar", "error": END},
    )

    graph.add_edge("search_similar", "analyze_photos")
    graph.add_edge("analyze_photos", "evaluate_price")
    graph.add_edge("evaluate_price", "generate_recommendation")
    graph.add_edge("generate_recommendation", "human_review")
    graph.add_edge("human_review", END)

    return graph.compile()


compiled_graph = build_graph()


def run_agent(listing_url: str) -> AgentState:
    initial_state: AgentState = {
        "listing_url": listing_url,
        "listing_data": None,
        "similar_listings": [],
        "price_stats": {},
        "photo_analysis": None,
        "price_evaluation": None,
        "recommendation": None,
        "user_feedback": None,
        "awaiting_feedback": False,
        "iteration": 0,
        "error": None,
    }
    return compiled_graph.invoke(initial_state)


def answer_followup(state: AgentState, question: str) -> AgentState:
    """Отвечает на уточняющий вопрос без повторного запуска всего пайплайна."""
    updated = {**state, "user_feedback": question, "awaiting_feedback": False}
    return compiled_graph.invoke(updated)
