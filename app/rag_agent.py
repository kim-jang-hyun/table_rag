"""LangGraph-based RAG agent.

Architecture::

    사용자 질문
        └─► retrieve_node  (Qdrant 벡터 검색)
                └─► grade_node  (문서 품질 평가)
                        ├─► [sufficient]   generate_node → 최종 응답  (rag_generate)
                        └─► [insufficient] web_rag_node  → 최종 응답  (web_fallback)

All questions are first searched against the vector DB; only when retrieved
documents are deemed insufficient does the agent fall back to Tavily web search.

Note
----
``router_node`` / ``route_decision`` are defined here for potential future use
but are not wired into the compiled graph.  The graph always starts at
``retrieve_node``.
"""

from __future__ import annotations

import logging
import os
from typing import List

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

import table_rag
from table_rag.models import get_llm

# ── Performance logger ────────────────────────────────────────────────────────

_perf_logger = logging.getLogger("table_rag.perf")
if not _perf_logger.handlers:
    _perf_logger.setLevel(logging.DEBUG)
    _ch = logging.StreamHandler()
    _ch.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    _perf_logger.addHandler(_ch)
    if not os.environ.get("STREAMLIT_SERVER_HEADLESS"):
        _fh = logging.FileHandler("perf.log", encoding="utf-8")
        _fh.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        _perf_logger.addHandler(_fh)

load_dotenv()


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str
    route: str          # "rag" | "general"
    documents: list     # search_and_rerank() result dicts
    grade: str          # "sufficient" | "insufficient"
    answer: str
    source: str         # "rag_generate" | "web_fallback"
    hybrid_warning: str
    collection: str
    use_reranker: bool
    use_hybrid: bool
    qdrant_top_k: int
    rerank_top_k: int
    openai_model: str


# ── Nodes ─────────────────────────────────────────────────────────────────────

ROUTER_SYSTEM = """당신은 사용자 질문을 분류하는 라우터입니다.

분류 기준:
- "rag": 업로드된 문서(PDF, PPT 등 공시, 보고서, 계약서, 재무제표 등 특정 문서)의 내용을 묻는 질문
- "general": 일반 상식, 최신 뉴스, 인터넷 검색이 필요한 질문, 또는 특정 문서 없이 답할 수 있는 질문

반드시 아래 JSON만 출력하세요. 다른 텍스트는 포함하지 마세요.
{{"route": "rag"}} 또는 {{"route": "general"}}"""


def router_node(state: AgentState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM),
        ("human", "{question}"),
    ])
    llm = ChatOpenAI(model=state.get("openai_model", table_rag.DEFAULT_OPENAI_MODEL), temperature=0)
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"question": state["question"]})
    route = result.get("route", "general")
    if route not in ("rag", "general"):
        route = "general"
    return {"route": route}


def route_decision(state: AgentState) -> str:
    return state["route"]


def retrieve_node(state: AgentState) -> dict:
    import time as _time

    _perf_logger.info("[retrieve_node] 시작")
    t0 = _time.time()
    docs, hybrid_warning = table_rag.search_and_rerank(
        query=state["question"],
        collection=state.get("collection", table_rag.DEFAULT_COLLECTION),
        qdrant_top_k=state.get("qdrant_top_k", 5),
        rerank_top_k=state.get("rerank_top_k", 5),
        use_reranker=state.get("use_reranker", True),
        use_hybrid=state.get("use_hybrid", True),
    )
    _perf_logger.info(f"[retrieve_node] 완료 ({_time.time() - t0:.1f}s) — {len(docs)}개 문서")
    return {"documents": docs, "hybrid_warning": hybrid_warning}


GRADE_SYSTEM = """당신은 검색된 문서가 사용자 질문에 충분히 답할 수 있는지 평가하는 평가자입니다.

평가 기준:
- "sufficient": 검색된 문서 중 질문과 관련된 내용이 있어 답변 생성이 가능한 경우
- "insufficient": 검색된 문서가 없거나, 질문과 관련 없는 내용만 있어 답변 불가한 경우

반드시 아래 JSON만 출력하세요. 다른 텍스트는 포함하지 마세요.
{{"grade": "sufficient"}} 또는 {{"grade": "insufficient"}}"""


def grade_node(state: AgentState) -> dict:
    import time as _time

    _perf_logger.info("[grade_node] 시작")
    t0 = _time.time()
    docs = state.get("documents", [])
    if not docs:
        _perf_logger.info("[grade_node] 문서 없음 → insufficient")
        return {"grade": "insufficient"}

    context_preview = "\n".join(d.get("text", "")[:300] for d in docs[:3])
    prompt = ChatPromptTemplate.from_messages([
        ("system", GRADE_SYSTEM),
        ("human", "질문: {question}\n\n검색된 문서 발췌:\n{context}"),
    ])
    llm = ChatOpenAI(model=state.get("openai_model", table_rag.DEFAULT_OPENAI_MODEL), temperature=0)
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"question": state["question"], "context": context_preview})
    grade = result.get("grade", "insufficient")
    if grade not in ("sufficient", "insufficient"):
        grade = "insufficient"
    _perf_logger.info(f"[grade_node] 완료 ({_time.time() - t0:.1f}s) — {grade}")
    return {"grade": grade}


def grade_decision(state: AgentState) -> str:
    return state["grade"]


def generate_node(state: AgentState) -> dict:
    import time as _time

    _perf_logger.info("[generate_node] 시작")
    t0 = _time.time()
    llm = get_llm(model=state.get("openai_model", table_rag.DEFAULT_OPENAI_MODEL))
    answer = table_rag.answer_with_openai(
        query=state["question"],
        docs=state.get("documents", []),
        llm=llm,
        use_reranker=state.get("use_reranker", True),
    )
    _perf_logger.info(f"[generate_node] 완료 ({_time.time() - t0:.1f}s)")
    return {"answer": answer, "source": "rag_generate"}


def _build_web_context(results: List[dict]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        url = r.get("url", "")
        content = r.get("content", "")
        lines.append(f"[{i}] {url}\n{content}")
    return "\n\n".join(lines)


WEB_RAG_SYSTEM = """당신은 웹 검색 결과를 바탕으로 질문에 답하는 AI 어시스턴트입니다.
반드시 제공된 검색 결과 내용을 근거로 답하세요.
검색 결과에 없는 내용은 추측하지 말고 모른다고 말하세요.
답변 끝에 참고한 출처 URL을 표시하세요."""


def web_rag_node(state: AgentState) -> dict:
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    tool = TavilySearchResults(max_results=5, tavily_api_key=tavily_key)
    results = tool.invoke(state["question"])

    context = _build_web_context(results) if results else ""
    if not context:
        return {"answer": "웹 검색 결과를 가져오지 못했습니다. 잠시 후 다시 시도하세요.", "source": "web_direct"}

    prompt = ChatPromptTemplate.from_messages([
        ("system", WEB_RAG_SYSTEM),
        ("human", "질문:\n{question}\n\n웹 검색 결과:\n{context}\n\n한국어로 간결하고 정확하게 답하세요."),
    ])
    llm = ChatOpenAI(model=state.get("openai_model", table_rag.DEFAULT_OPENAI_MODEL), temperature=0)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": state["question"], "context": context})
    return {"answer": answer, "source": "web_fallback"}


# ── Graph compilation ─────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("grade_node", grade_node)
    workflow.add_node("generate_node", generate_node)
    workflow.add_node("web_rag_node", web_rag_node)

    workflow.add_edge(START, "retrieve_node")
    workflow.add_edge("retrieve_node", "grade_node")

    workflow.add_conditional_edges(
        "grade_node",
        grade_decision,
        {
            "sufficient": "generate_node",
            "insufficient": "web_rag_node",
        },
    )

    workflow.add_edge("generate_node", END)
    workflow.add_edge("web_rag_node", END)

    return workflow


agent = _build_graph().compile()


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    demos = [
        "포스코홀딩스 주식 소각 결정의 주요 내용은 무엇인가요?",
        "2026년 최신 AI 트렌드는 무엇인가요?",
    ]

    for q in demos:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        result = agent.invoke(
            {
                "question": q,
                "route": "",
                "documents": [],
                "grade": "",
                "answer": "",
                "source": "",
                "hybrid_warning": "",
                "collection": table_rag.DEFAULT_COLLECTION,
                "use_reranker": True,
                "use_hybrid": True,
                "qdrant_top_k": 20,
                "rerank_top_k": 5,
                "openai_model": table_rag.DEFAULT_OPENAI_MODEL,
            }
        )
        print(f"source: {result['source']}")
        print(f"A: {result['answer'][:300]}...")
