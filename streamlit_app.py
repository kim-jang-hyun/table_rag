import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
def _embed_model_cached():
    return basic_rag._load_embed_model()


@st.cache_resource
def _reranker_cached():
    return basic_rag._load_reranker()


@st.cache_resource
def _clients_cached() -> Tuple[Any, Any]:
    qdrant = basic_rag._get_qdrant_client()
    openai_client = basic_rag._get_openai_client()
    return qdrant, openai_client


def _list_local_pdfs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("*.pdf") if p.is_file()])


def _ingest_pdfs(
    *,
    pdf_paths: Sequence[Path],
    collection: str,
    extract_table_chunks: bool,
) -> int:
    embed_model = _embed_model_cached()
    return basic_rag.ingest_pdfs_to_qdrant(
        pdf_paths=list(pdf_paths),
        collection=collection,
        embed_model=embed_model,
        extract_table_chunks=extract_table_chunks,
    )


def _ask(
    *,
    question: str,
    collection: str,
    openai_model: str,
    qdrant_top_k: int,
    rerank_top_k: int,
    use_reranker: bool,
) -> Dict[str, Any]:
    embed_model = _embed_model_cached()
    reranker = _reranker_cached() if use_reranker else None
    qdrant, openai_client = _clients_cached()

    docs = basic_rag.search_and_rerank(
        query=question,
        collection=collection,
        qdrant_top_k=qdrant_top_k,
        rerank_top_k=rerank_top_k,
        use_reranker=use_reranker,
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
            use_reranker=use_reranker,
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


def _build_pdf_context_text(*, pdf_path: Path, max_chars: int, extract_table_chunks: bool) -> str:
    chunks = basic_rag._load_pdf_sections(pdf_path, extract_table_chunks=extract_table_chunks)
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


def _build_pdf_context_text_many(
    *, pdf_paths: Sequence[Path], max_chars: int, extract_table_chunks: bool
) -> str:
    blocks: List[str] = []
    for p in pdf_paths:
        body = _build_pdf_context_text(
            pdf_path=p, max_chars=0, extract_table_chunks=extract_table_chunks
        )
        if body:
            blocks.append(f"### 문서: {p.name}\n{body}")
    combined = "\n\n".join(blocks).strip()
    if not combined:
        return ""
    if max_chars and len(combined) > max_chars:
        combined = combined[:max_chars].rstrip() + "\n\n...(truncated)"
    return combined


def _dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    seen: set = set()
    out: List[Path] = []
    for p in paths:
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _default_selected_pdf_names(names: List[str]) -> List[str]:
    if not names:
        return []
    raw = os.environ.get("PDF_PATHS", "").strip()
    picked: List[str] = []
    if raw:
        for part in re.split(r"[,;\n]+", raw):
            stem = Path(part.strip()).name
            if stem in names:
                picked.append(stem)
    default_pdf = os.environ.get("PDF_PATH", basic_rag.DEFAULT_PDF)
    if not picked and default_pdf in names:
        picked = [default_pdf]
    if not picked:
        picked = [names[0]]
    return [n for n in picked if n in names]


def _ask_llm_with_pdf_text(
    *,
    question: str,
    openai_model: str,
    pdf_paths: Sequence[Path],
    max_context_chars: int,
    extract_table_chunks: bool,
) -> Dict[str, Any]:
    _, openai_client = _clients_cached()
    paths = list(pdf_paths)
    if len(paths) == 1:
        context = _build_pdf_context_text(
            pdf_path=paths[0],
            max_chars=max_context_chars,
            extract_table_chunks=extract_table_chunks,
        )
    else:
        context = _build_pdf_context_text_many(
            pdf_paths=paths,
            max_chars=max_context_chars,
            extract_table_chunks=extract_table_chunks,
        )
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
    st.caption(
        "PDF를 여러 개 선택해 한 컬렉션에 인덱싱할 수 있습니다. 질문 시 Qdrant 검색 후, 옵션에 따라 리랭커를 거쳐 OpenAI로 답변합니다."
    )

    _ensure_env_from_sidebar()

    with st.sidebar:
        st.subheader("RAG 설정")
        use_rag = st.toggle("RAG 사용 (검색+선택적 rerank)", value=True)
        collection = st.text_input("Qdrant collection", value=os.environ.get("QDRANT_COLLECTION", basic_rag.DEFAULT_COLLECTION))
        openai_model = st.text_input("OpenAI model", value=os.environ.get("OPENAI_MODEL", basic_rag.DEFAULT_OPENAI_MODEL))
        qdrant_top_k = st.slider("Qdrant top_k", min_value=5, max_value=50, value=20, step=1, disabled=not use_rag)
        _env_rerank = os.environ.get("USE_RERANKER", "1").strip().lower() not in {"0", "false", "no"}
        use_reranker = st.toggle(
            "리랭커 사용 (bge-reranker-v2-m3)",
            value=_env_rerank,
            disabled=not use_rag,
            help="끄면 Qdrant 벡터 유사도만으로 상위 청크를 고릅니다. 리랭커 모델을 로드하지 않아 메모리를 덜 씁니다.",
        )
        rerank_top_k = st.slider(
            "최종 컨텍스트 청크 수",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            disabled=not use_rag,
            help="리랭커를 켠 경우: rerank 후 이 개수만 LLM에 전달. 끈 경우: 벡터 검색 상위 이 개수.",
        )
        extract_table_chunks = st.toggle(
            "테이블 전용 청크 추출",
            value=True,
            help="끄면 구조화된 표 청크(source_type=table)는 만들지 않고, 페이지 전체 텍스트 청크만 사용합니다. 변경 후 인덱싱을 다시 실행하세요.",
        )

        st.subheader("PDF")
        pdf_folder = Path(os.environ.get("PDF_FOLDER", ".")).resolve()
        pdfs = _list_local_pdfs(pdf_folder)
        default_pdf = os.environ.get("PDF_PATH", basic_rag.DEFAULT_PDF)

        folder_paths: List[Path] = []
        if pdfs:
            names = [p.name for p in pdfs]
            default_names = _default_selected_pdf_names(names)
            selected_names = st.multiselect(
                "로컬 PDF 선택 (복수 가능)",
                options=names,
                default=default_names,
                help="같은 collection에 선택한 모든 PDF를 함께 인덱싱합니다.",
            )
            folder_paths = [pdf_folder / n for n in selected_names]
        else:
            st.info("현재 폴더에서 PDF를 찾지 못했어요. 아래 업로더를 사용하거나, PDF를 프로젝트 폴더에 두세요.")
            if not Path(default_pdf).exists():
                st.caption(f"단일 경로 힌트: `PDF_PATH={default_pdf}` (파일이 없으면 업로드만 사용)")

        upload_dir = Path(".streamlit_uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        uploaded_list = st.file_uploader(
            "또는 PDF 업로드 (복수 가능)",
            type=["pdf"],
            accept_multiple_files=True,
        )
        upload_paths: List[Path] = []
        if uploaded_list:
            for up in uploaded_list:
                dest = upload_dir / up.name
                dest.write_bytes(up.getbuffer())
                upload_paths.append(dest)
            st.success(f"업로드 완료: {len(upload_paths)}개 파일")

        fallback: List[Path] = []
        if not folder_paths and not upload_paths:
            p = Path(default_pdf)
            if p.exists():
                fallback = [p]

        pdf_paths = _dedupe_paths(list(folder_paths) + upload_paths + fallback)

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
            if not pdf_paths:
                st.error("인덱싱할 PDF가 없습니다. 폴더에서 선택하거나 파일을 업로드하세요.")
            else:
                missing = [str(p) for p in pdf_paths if not p.exists()]
                if missing:
                    st.error("PDF를 찾을 수 없습니다:\n" + "\n".join(missing))
                else:
                    with st.spinner(
                        f"{len(pdf_paths)}개 PDF를 읽고 임베딩한 뒤 Qdrant에 업서트 중..."
                    ):
                        n_chunks = _ingest_pdfs(
                            pdf_paths=pdf_paths,
                            collection=collection,
                            extract_table_chunks=extract_table_chunks,
                        )
                    doc_list = ", ".join(p.name for p in pdf_paths)
                    st.success(
                        f"인덱싱 완료. PDF {len(pdf_paths)}개 · 청크 {n_chunks:,}개 · collection: `{collection}`\n\n{doc_list}"
                    )
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
                spin = "검색 + rerank + 답변 생성 중..." if use_reranker else "검색(벡터) + 답변 생성 중..."
                with st.spinner(spin):
                    result = _ask(
                        question=question,
                        collection=collection,
                        openai_model=openai_model,
                        qdrant_top_k=qdrant_top_k,
                        rerank_top_k=rerank_top_k,
                        use_reranker=use_reranker,
                    )
            else:
                if inject_pdf_text:
                    usable = [p for p in pdf_paths if p.exists()]
                    if not usable:
                        result = {
                            "answer": "PDF를 찾을 수 없습니다. 사이드바에서 파일을 선택하거나 업로드하세요.",
                            "docs": [],
                        }
                    else:
                        with st.spinner("LLM 답변 생성 중 (비RAG: PDF 텍스트 직접 주입)..."):
                            result = _ask_llm_with_pdf_text(
                                question=question,
                                openai_model=openai_model,
                                pdf_paths=usable,
                                max_context_chars=max_context_chars,
                                extract_table_chunks=extract_table_chunks,
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
                        rank_label = "rerank" if (use_rag and use_reranker) else "vector"
                        for i, d in enumerate(docs, start=1):
                            doc_label = d.get("doc") or ""
                            doc_bit = f"  doc=`{doc_label}`" if doc_label else ""
                            st.markdown(
                                f"**[{i}]**{doc_bit}  page={d.get('page')}  chunk_id={d.get('chunk_id')}  "
                                f"type={d.get('source_type')}  {rank_label}={d.get('rerank_score', 0.0):.4f}"
                            )
                            st.code(d.get("text", ""), language="text")
        except Exception as e:
            st.exception(e)


if __name__ == "__main__":
    main()

