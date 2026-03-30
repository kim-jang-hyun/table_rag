import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

import basic_rag


st.set_page_config(page_title="Table RAG", page_icon="📄", layout="wide")


def _load_env() -> None:
    # Single source of truth: .env (local dev)
    load_dotenv(override=False)


def _ensure_env_from_sidebar() -> None:
    with st.sidebar:
        st.subheader("연결 설정")

        qdrant_url = st.text_input(
            "QDRANT_URL",
            value=os.environ.get("QDRANT_URL", ""),
            placeholder="https://...:6333",
        )
        qdrant_api_key = st.text_input(
            "QDRANT_API_KEY",
            value=os.environ.get("QDRANT_API_KEY", ""),
            type="password",
        )
        openai_api_key = st.text_input(
            "OPENAI_API_KEY",
            value=os.environ.get("OPENAI_API_KEY", ""),
            type="password",
        )

        if qdrant_url.strip():
            os.environ["QDRANT_URL"] = qdrant_url.strip()
        if qdrant_api_key.strip():
            os.environ["QDRANT_API_KEY"] = qdrant_api_key.strip()
        if openai_api_key.strip():
            os.environ["OPENAI_API_KEY"] = openai_api_key.strip()


@st.cache_resource
def _load_models_cached():
    return basic_rag._load_models()


@st.cache_resource
def _clients_cached() -> Tuple[Any, Any]:
    qdrant = basic_rag._get_qdrant_client()
    openai_client = basic_rag._get_openai_client()
    return qdrant, openai_client


def _list_local_pdfs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("*.pdf") if p.is_file()])


def _ingest(
    *,
    pdf_path: Path,
    collection: str,
) -> int:
    embed_model, _ = _load_models_cached()
    basic_rag.ingest_pdf_to_qdrant(pdf_path=pdf_path, collection=collection, embed_model=embed_model)
    chunks = basic_rag._load_pdf_sections(pdf_path)
    return len(chunks)


def _ask(
    *,
    question: str,
    collection: str,
    openai_model: str,
    qdrant_top_k: int,
    rerank_top_k: int,
) -> Dict[str, Any]:
    embed_model, reranker = _load_models_cached()
    qdrant, openai_client = _clients_cached()

    docs = basic_rag.search_and_rerank(
        query=question,
        collection=collection,
        qdrant_top_k=qdrant_top_k,
        rerank_top_k=rerank_top_k,
        embed_model=embed_model,
        reranker=reranker,
        qdrant=qdrant,
    )

    answer = ""
    if docs:
        answer = basic_rag.answer_with_openai(
            query=question,
            docs=docs,
            openai_client=openai_client,
            model=openai_model,
        )

    return {"answer": answer, "docs": docs}


def _ask_llm_only(*, question: str, openai_model: str) -> Dict[str, Any]:
    _, openai_client = _clients_cached()
    system_prompt = (
        "당신은 유능한 AI 어시스턴트입니다. 아래 질문에 한국어로 간결하고 정확하게 답하세요. "
        "문서 컨텍스트나 검색 결과는 제공되지 않았습니다. 확실하지 않으면 모른다고 말하세요."
    )
    response = openai_client.responses.create(
        model=openai_model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.1,
    )
    return {"answer": (response.output_text or "").strip(), "docs": []}


def _build_pdf_context_text(*, pdf_path: Path, max_chars: int) -> str:
    chunks = basic_rag._load_pdf_sections(pdf_path)
    if not chunks:
        return ""
    parts: List[str] = []
    for c in chunks:
        parts.append(f"[page={c.page} type={c.source_type} chunk_id={c.chunk_id}]")
        parts.append(c.text)
        parts.append("")
    context = "\n".join(parts).strip()
    if max_chars and len(context) > max_chars:
        context = context[:max_chars].rstrip() + "\n\n...(truncated)"
    return context


def _ask_llm_with_pdf_text(
    *,
    question: str,
    openai_model: str,
    pdf_path: Path,
    max_context_chars: int,
) -> Dict[str, Any]:
    _, openai_client = _clients_cached()
    context = _build_pdf_context_text(pdf_path=pdf_path, max_chars=max_context_chars)
    if not context:
        return {"answer": "PDF에서 텍스트를 추출하지 못했습니다. (스캔 이미지 PDF일 수 있어요)", "docs": []}

    system_prompt = (
        "당신은 문서 QA 어시스턴트입니다. 반드시 제공된 컨텍스트 범위 안에서만 답하세요. "
        "근거가 부족하면 모른다고 말하고, 추측하지 마세요."
    )
    user_prompt = f"질문:\n{question}\n\n컨텍스트(문서 발췌):\n{context}\n\n요청: 한국어로 간결하고 정확하게 답하세요."
    response = openai_client.responses.create(
        model=openai_model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return {"answer": (response.output_text or "").strip(), "docs": []}


def main() -> None:
    _load_env()

    st.title("📄 Table RAG (PDF → Qdrant → OpenAI)")
    st.caption("PDF를 인덱싱하고 질문하면, Qdrant 검색 + rerank 후 OpenAI로 답변합니다.")

    _ensure_env_from_sidebar()

    with st.sidebar:
        st.subheader("RAG 설정")
        use_rag = st.toggle("RAG 사용 (검색+rerank)", value=True)
        collection = st.text_input("Qdrant collection", value=os.environ.get("QDRANT_COLLECTION", basic_rag.DEFAULT_COLLECTION))
        openai_model = st.text_input("OpenAI model", value=os.environ.get("OPENAI_MODEL", basic_rag.DEFAULT_OPENAI_MODEL))
        qdrant_top_k = st.slider("Qdrant top_k", min_value=5, max_value=50, value=20, step=1, disabled=not use_rag)
        rerank_top_k = st.slider("Rerank top_k", min_value=1, max_value=20, value=5, step=1, disabled=not use_rag)

        st.subheader("PDF")
        pdf_folder = Path(os.environ.get("PDF_FOLDER", ".")).resolve()
        pdfs = _list_local_pdfs(pdf_folder)
        default_pdf = os.environ.get("PDF_PATH", basic_rag.DEFAULT_PDF)

        if pdfs:
            names = [p.name for p in pdfs]
            initial = names.index(default_pdf) if default_pdf in names else 0
            pdf_name = st.selectbox("로컬 PDF 선택", options=names, index=initial)
            pdf_path = pdf_folder / pdf_name
        else:
            pdf_path = Path(default_pdf)
            st.info("현재 폴더에서 PDF를 찾지 못했어요. 아래 업로더를 사용하거나, PDF를 프로젝트 폴더에 두세요.")

        uploaded = st.file_uploader("또는 PDF 업로드", type=["pdf"])
        if uploaded is not None:
            upload_dir = Path(".streamlit_uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = upload_dir / uploaded.name
            pdf_path.write_bytes(uploaded.getbuffer())
            st.success(f"업로드 완료: {pdf_path.name}")

        st.subheader("비RAG(문서 직접 주입) 설정")
        inject_pdf_text = st.toggle("RAG 꺼도 PDF 텍스트를 LLM에 넣기", value=False, disabled=use_rag)
        max_context_chars = st.slider(
            "PDF 컨텍스트 최대 글자 수",
            min_value=2_000,
            max_value=120_000,
            value=20_000,
            step=1_000,
            disabled=use_rag or (not inject_pdf_text),
        )

        ingest_clicked = st.button(
            "인덱싱(업서트) 실행",
            type="primary",
            use_container_width=True,
            disabled=not use_rag,
        )

    if ingest_clicked and use_rag:
        try:
            if not pdf_path.exists():
                st.error(f"PDF를 찾을 수 없습니다: {pdf_path}")
            else:
                with st.spinner("PDF를 읽고 임베딩한 뒤 Qdrant에 업서트 중..."):
                    n_chunks = _ingest(pdf_path=pdf_path, collection=collection)
                st.success(f"인덱싱 완료. 추출 청크 수: {n_chunks:,}  |  collection: {collection}")
        except Exception as e:
            st.exception(e)

    st.divider()

    if "chat" not in st.session_state:
        st.session_state.chat = []  # type: ignore[attr-defined]

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("질문을 입력하세요")
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            if use_rag:
                with st.spinner("검색 + rerank + 답변 생성 중..."):
                    result = _ask(
                        question=question,
                        collection=collection,
                        openai_model=openai_model,
                        qdrant_top_k=qdrant_top_k,
                        rerank_top_k=rerank_top_k,
                    )
            else:
                if inject_pdf_text:
                    if not pdf_path.exists():
                        result = {"answer": f"PDF를 찾을 수 없습니다: {pdf_path}", "docs": []}
                    else:
                        with st.spinner("LLM 답변 생성 중 (비RAG: PDF 텍스트 직접 주입)..."):
                            result = _ask_llm_with_pdf_text(
                                question=question,
                                openai_model=openai_model,
                                pdf_path=pdf_path,
                                max_context_chars=max_context_chars,
                            )
                else:
                    with st.spinner("LLM 답변 생성 중 (RAG 미사용: 문서 없음)..."):
                        result = _ask_llm_only(question=question, openai_model=openai_model)
            answer = result["answer"] or "검색 결과가 없거나 답변 생성에 실패했습니다."
            docs = result["docs"] or []

            st.session_state.chat.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
                if docs:
                    with st.expander(f"근거 청크 보기 (top {len(docs)})", expanded=False):
                        for i, d in enumerate(docs, start=1):
                            st.markdown(
                                f"**[{i}]** page={d.get('page')}  chunk_id={d.get('chunk_id')}  "
                                f"type={d.get('source_type')}  rerank={d.get('rerank_score', 0.0):.4f}"
                            )
                            st.code(d.get("text", ""), language="text")
        except Exception as e:
            st.exception(e)


if __name__ == "__main__":
    main()

