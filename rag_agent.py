"""
LangGraph 기반 RAG 에이전트

아키텍처:
  사용자 질문
      └─► router_node (RAG 주제 vs 일반 주제)
              ├─► [RAG 경로] retrieve_node → grade_node
              │       ├─► [충분] generate_node → 최종 응답
              │       └─► [부족] web_rag_node  → 최종 응답  (폴백)
              └─► [일반 경로] web_rag_node      → 최종 응답  (직접 호출)

일반 주제 / 문서 부족 시 Tavily 웹검색 결과를 컨텍스트로 LLM에 입력(Web RAG).
"""

from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

import basic_rag

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────

def _llm(model: str = basic_rag.DEFAULT_OPENAI_MODEL) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=0)


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str
    route: str          # "rag" | "general"
    documents: list     # basic_rag.search_and_rerank() 반환 dict 리스트
    grade: str          # "sufficient" | "insufficient"
    answer: str
    source: str         # "rag_generate" | "web_direct" | "web_fallback"
    hybrid_warning: str
    # 호출자가 주입하는 검색 설정
    collection: str
    use_reranker: bool
    use_hybrid: bool
    qdrant_top_k: int
    rerank_top_k: int
    openai_model: str


# ── 노드 ──────────────────────────────────────────────────────────────────────

ROUTER_SYSTEM = """당신은 사용자 질문을 분류하는 라우터입니다.

분류 기준:
- "rag": 업로드된 PDF 문서(공시, 보고서, 계약서, 재무제표 등 특정 문서)의 내용을 묻는 질문
- "general": 일반 상식, 최신 뉴스, 인터넷 검색이 필요한 질문, 또는 특정 문서 없이 답할 수 있는 질문

반드시 아래 JSON만 출력하세요. 다른 텍스트는 포함하지 마세요.
{{"route": "rag"}} 또는 {{"route": "general"}}"""


def router_node(state: AgentState) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_SYSTEM),
        ("human", "{question}"),
    ])
    llm = _llm(state.get("openai_model", basic_rag.DEFAULT_OPENAI_MODEL))
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({"question": state["question"]})
    route = result.get("route", "general")
    if route not in ("rag", "general"):
        route = "general"
    return {"route": route}


def route_decision(state: AgentState) -> str:
    return state["route"]


def retrieve_node(state: AgentState) -> dict:
    docs, hybrid_warning = basic_rag.search_and_rerank(
        query=state["question"],
        collection=state.get("collection", basic_rag.DEFAULT_COLLECTION),
        qdrant_top_k=state.get("qdrant_top_k", 20),
        rerank_top_k=state.get("rerank_top_k", 5),
        use_reranker=state.get("use_reranker", True),
        use_hybrid=state.get("use_hybrid", True),
    )
    return {"documents": docs, "hybrid_warning": hybrid_warning}


GRADE_SYSTEM = """당신은 검색된 문서가 사용자 질문에 충분히 답할 수 있는지 평가하는 평가자입니다.

평가 기준:
- "sufficient": 검색된 문서 중 질문과 관련된 내용이 있어 답변 생성이 가능한 경우
- "insufficient": 검색된 문서가 없거나, 질문과 관련 없는 내용만 있어 답변 불가한 경우

반드시 아래 JSON만 출력하세요. 다른 텍스트는 포함하지 마세요.
{{"grade": "sufficient"}} 또는 {{"grade": "insufficient"}}"""


def grade_node(state: AgentState) -> dict:
    docs = state.get("documents", [])
    if not docs:
        return {"grade": "insufficient"}

    context_preview = "\n".join(
        d.get("text", "")[:300] for d in docs[:3]
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", GRADE_SYSTEM),
        ("human", "질문: {question}\n\n검색된 문서 발췌:\n{context}"),
    ])
    llm = _llm(state.get("openai_model", basic_rag.DEFAULT_OPENAI_MODEL))
    chain = prompt | llm | JsonOutputParser()
    result = chain.invoke({
        "question": state["question"],
        "context": context_preview,
    })
    grade = result.get("grade", "insufficient")
    if grade not in ("sufficient", "insufficient"):
        grade = "insufficient"
    return {"grade": grade}


def grade_decision(state: AgentState) -> str:
    return state["grade"]


def generate_node(state: AgentState) -> dict:
    llm = basic_rag._get_llm(
        model=state.get("openai_model", basic_rag.DEFAULT_OPENAI_MODEL)
    )
    answer = basic_rag.answer_with_openai(
        query=state["question"],
        docs=state.get("documents", []),
        llm=llm,
        use_reranker=state.get("use_reranker", True),
    )
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
        answer = "웹 검색 결과를 가져오지 못했습니다. 잠시 후 다시 시도하세요."
        source = state.get("source", "web_direct")
        return {"answer": answer, "source": source}

    prompt = ChatPromptTemplate.from_messages([
        ("system", WEB_RAG_SYSTEM),
        ("human", "질문:\n{question}\n\n웹 검색 결과:\n{context}\n\n한국어로 간결하고 정확하게 답하세요."),
    ])
    llm = _llm(state.get("openai_model", basic_rag.DEFAULT_OPENAI_MODEL))
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": state["question"], "context": context})

    # router에서 "general"로 왔으면 web_direct, grade에서 왔으면 web_fallback
    source = "web_fallback" if state.get("route") == "rag" else "web_direct"
    return {"answer": answer, "source": source}


# ── 그래프 컴파일 ─────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("router_node", router_node)
    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("grade_node", grade_node)
    workflow.add_node("generate_node", generate_node)
    workflow.add_node("web_rag_node", web_rag_node)

    workflow.add_edge(START, "router_node")

    workflow.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "rag": "retrieve_node",
            "general": "web_rag_node",
        },
    )

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


# ── CLI 테스트 ────────────────────────────────────────────────────────────────

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
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = agent.invoke({
            "question": q,
            "route": "",
            "documents": [],
            "grade": "",
            "answer": "",
            "source": "",
            "hybrid_warning": "",
            "collection": basic_rag.DEFAULT_COLLECTION,
            "use_reranker": True,
            "use_hybrid": True,
            "qdrant_top_k": 20,
            "rerank_top_k": 5,
            "openai_model": basic_rag.DEFAULT_OPENAI_MODEL,
        })
        print(f"route : {result['route']}")
        print(f"source: {result['source']}")
        print(f"A: {result['answer'][:300]}...")
