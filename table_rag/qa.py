"""LLM-based answer generation using retrieved document context."""

from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import DEFAULT_OPENAI_MODEL


def build_context_from_docs(docs: List[dict], *, use_reranker: bool = True) -> str:
    """Format retrieved chunks into a numbered context string for the LLM prompt.

    Args:
        docs:         List of result dicts from :func:`~table_rag.retrieval.search_and_rerank`.
        use_reranker: Controls score label (``"rerank"`` vs ``"vector"``).

    Returns:
        Multi-line string with one numbered section per chunk.
    """
    lines: List[str] = []
    rank_key = "rerank" if use_reranker else "vector"
    for i, doc in enumerate(docs, start=1):
        doc_name = doc.get("doc") or ""
        doc_bit = f" doc={doc_name}" if doc_name else ""
        lines.append(
            f"[{i}]{doc_bit} page={doc.get('page')} chunk_id={doc.get('chunk_id')} "
            f"type={doc.get('source_type')} {rank_key}={doc.get('rerank_score', 0.0):.4f}"
        )
        lines.append(doc.get("text", ""))
        lines.append("")
    return "\n".join(lines).strip()


def answer_with_openai(
    *,
    query: str,
    docs: List[dict],
    llm: ChatOpenAI,
    model: str = DEFAULT_OPENAI_MODEL,
    use_reranker: bool = True,
) -> str:
    """Generate a Korean answer from retrieved chunks using an OpenAI LLM.

    The LLM is instructed to answer only within the provided context and to
    cite chunk numbers at the end of its response.

    Args:
        query:        The user's question.
        docs:         Retrieved chunks from :func:`~table_rag.retrieval.search_and_rerank`.
        llm:          A :class:`langchain_openai.ChatOpenAI` instance.
        model:        Model name string (informational only; the actual model is
                      determined by *llm*).
        use_reranker: Passed to :func:`build_context_from_docs` for score labeling.

    Returns:
        The LLM-generated answer string, or an error message when context is empty.
    """
    context = build_context_from_docs(docs, use_reranker=use_reranker)
    if not context:
        return "검색된 컨텍스트가 없어 답변을 생성할 수 없습니다."

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 문서 QA 어시스턴트입니다. 반드시 제공된 컨텍스트 범위 안에서만 답하세요. "
                "근거가 부족하면 모른다고 말하고, 추측하지 마세요. "
                "가능하면 답변 끝에 근거 청크 번호([1], [2]...)를 표시하세요.",
            ),
            (
                "human",
                "질문:\n{query}\n\n컨텍스트:\n{context}\n\n"
                "요청: 질문에 한국어로 간결하고 정확하게 답하세요.",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query, "context": context})
