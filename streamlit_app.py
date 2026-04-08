import os
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

_APP_DIR = Path(__file__).parent

import streamlit as st
from dotenv import load_dotenv

import basic_rag
import rag_agent


st.set_page_config(page_title="Table RAG", page_icon="📄", layout="wide")

_SOURCE_LABELS: Dict[str, str] = {
    "rag_generate": "📄 RAG 검색 (벡터 DB)",
    "web_direct": "🌐 웹 검색 (직접 호출)",
    "web_fallback": "🌐 웹 검색 (폴백 — 문서 부족)",
}


def _load_env() -> None:
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
        tavily_api_key = st.text_input(
            "TAVILY_API_KEY",
            value=os.environ.get("TAVILY_API_KEY", ""),
            type="password",
        )

        if qdrant_url.strip():
            os.environ["QDRANT_URL"] = qdrant_url.strip()
        if qdrant_api_key.strip():
            os.environ["QDRANT_API_KEY"] = qdrant_api_key.strip()
        if openai_api_key.strip():
            os.environ["OPENAI_API_KEY"] = openai_api_key.strip()
        if tavily_api_key.strip():
            os.environ["TAVILY_API_KEY"] = tavily_api_key.strip()


@st.cache_resource
def _embed_model_cached():
    return basic_rag._load_embed_model()


@st.cache_resource
def _reranker_cached():
    return basic_rag._load_reranker()


@st.cache_resource
def _qdrant_cached():
    return basic_rag._get_qdrant_client()


def _list_local_pdfs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("*.pdf") if p.is_file()])


def _ingest_pdfs(
    *,
    pdf_paths: Sequence[Path],
    collection: str,
    extract_table_chunks: bool,
    enable_hybrid: bool,
    merge_cross_page_tables: bool,
) -> int:
    embed_model = _embed_model_cached()
    return basic_rag.ingest_pdfs_to_qdrant(
        pdf_paths=list(pdf_paths),
        collection=collection,
        embed_model=embed_model,
        extract_table_chunks=extract_table_chunks,
        enable_hybrid=enable_hybrid,
        merge_cross_page_tables=merge_cross_page_tables,
    )


def _ask_with_agent(
    *,
    question: str,
    collection: str,
    openai_model: str,
    qdrant_top_k: int,
    rerank_top_k: int,
    use_reranker: bool,
    use_hybrid: bool,
) -> Dict[str, Any]:
    initial_state: rag_agent.AgentState = {
        "question": question,
        "route": "",
        "documents": [],
        "grade": "",
        "answer": "",
        "source": "",
        "hybrid_warning": "",
        "collection": collection,
        "use_reranker": use_reranker,
        "use_hybrid": use_hybrid,
        "qdrant_top_k": qdrant_top_k,
        "rerank_top_k": rerank_top_k,
        "openai_model": openai_model,
    }
    result = rag_agent.agent.invoke(initial_state)
    return {
        "answer": result.get("answer", ""),
        "docs": result.get("documents", []),
        "source": result.get("source", ""),
        "route": result.get("route", ""),
        "grade": result.get("grade", ""),
        "hybrid_warning": result.get("hybrid_warning", ""),
    }


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
    if picked:
        return [n for n in picked if n in names]
    # PDF_FOLDER 기반으로 관리하는 경우: 폴더 내 전체 PDF를 기본 선택
    return list(names)


def main() -> None:
    _load_env()

    st.title("📄 Table RAG (PDF → Qdrant → LangGraph)")
    st.caption(
        "LangGraph 기반 라우팅: PDF 관련 질문은 벡터 DB 검색(RAG), "
        "일반/문서 부족 질문은 Tavily 웹검색(Web RAG)으로 자동 분기합니다."
    )

    _ensure_env_from_sidebar()

    with st.sidebar:
        st.subheader("RAG 설정")

        hybrid_ok = basic_rag.is_fastembed_available()
        if not hybrid_ok:
            st.caption(
                "참고: fastembed가 없어 BM25 하이브리드는 동작하지 않습니다. "
                "같은 venv에서 `pip install fastembed` 후 앱을 다시 실행하세요."
            )
        collection = st.text_input(
            "Qdrant collection",
            value=os.environ.get("QDRANT_COLLECTION", basic_rag.DEFAULT_COLLECTION),
        )
        openai_model = st.text_input(
            "OpenAI model",
            value=os.environ.get("OPENAI_MODEL", basic_rag.DEFAULT_OPENAI_MODEL),
        )
        qdrant_top_k = st.slider("Qdrant top_k", min_value=5, max_value=50, value=20, step=1)
        _env_rerank = os.environ.get("USE_RERANKER", "1").strip().lower() not in {"0", "false", "no"}
        use_reranker = st.toggle(
            "리랭커 사용 (bge-reranker-v2-m3)",
            value=_env_rerank,
            help="끄면 Qdrant 벡터 유사도만으로 상위 청크를 고릅니다.",
        )
        rerank_top_k = st.slider(
            "최종 컨텍스트 청크 수",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="리랭커를 켠 경우: rerank 후 이 개수만 LLM에 전달. 끈 경우: 벡터 상위 이 개수.",
        )
        _default_hybrid = os.environ.get("USE_HYBRID", "1").strip().lower() not in {"0", "false", "no"}
        use_hybrid = st.toggle(
            "하이브리드 (BM25 + 벡터)",
            value=_default_hybrid,
            help="켜면 dense+BM25 sparse를 RRF로 결합합니다. 변경 후 재인덱싱하세요.",
        )
        extract_table_chunks = st.toggle(
            "테이블 전용 청크 추출",
            value=True,
            help="끄면 페이지 전체 텍스트 청크만 사용합니다. 변경 후 재인덱싱하세요.",
        )
        _env_mcp = os.environ.get("MERGE_CROSS_PAGE_TABLES", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        merge_cross_page_tables = st.toggle(
            "페이지 간 표 병합",
            value=_env_mcp,
            disabled=not extract_table_chunks,
            help="켜면 여러 페이지에 걸친 표를 하나의 청크로 합칩니다. 변경 후 재인덱싱하세요.",
        )

        st.subheader("PDF")
        _pdf_folder_env = os.environ.get("PDF_FOLDER", "테스트문서")
        _pdf_folder_raw = Path(_pdf_folder_env)
        pdf_folder = (_pdf_folder_raw if _pdf_folder_raw.is_absolute() else _APP_DIR / _pdf_folder_raw).resolve()
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

        ingest_clicked = st.button(
            "인덱싱(업서트) 실행",
            type="primary",
            use_container_width=True,
        )

    if ingest_clicked:
        try:
            if not pdf_paths:
                st.error("인덱싱할 PDF가 없습니다. 폴더에서 선택하거나 파일을 업로드하세요.")
            else:
                missing = [str(p) for p in pdf_paths if not p.exists()]
                if missing:
                    st.error("PDF를 찾을 수 없습니다:\n" + "\n".join(missing))
                else:
                    spin = (
                        f"{len(pdf_paths)}개 PDF → 의미+키워드 임베딩 후 Qdrant 업서트..."
                        if use_hybrid
                        else f"{len(pdf_paths)}개 PDF → 의미 임베딩 후 Qdrant 업서트..."
                    )
                    with st.spinner(spin):
                        n_chunks = _ingest_pdfs(
                            pdf_paths=pdf_paths,
                            collection=collection,
                            extract_table_chunks=extract_table_chunks,
                            enable_hybrid=use_hybrid,
                            merge_cross_page_tables=(
                                merge_cross_page_tables and extract_table_chunks
                            ),
                        )
                    doc_list = ", ".join(p.name for p in pdf_paths)
                    mode = "하이브리드 (의미+키워드)" if use_hybrid else "의미 벡터만"
                    st.success(
                        f"인덱싱 완료 ({mode}). PDF {len(pdf_paths)}개 · 청크 {n_chunks:,}개 · collection: `{collection}`\n\n{doc_list}"
                    )
        except Exception as e:
            st.exception(e)

    st.divider()

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("질문을 입력하세요")
    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            with st.spinner("라우팅 및 답변 생성 중..."):
                result = _ask_with_agent(
                    question=question,
                    collection=collection,
                    openai_model=openai_model,
                    qdrant_top_k=qdrant_top_k,
                    rerank_top_k=rerank_top_k,
                    use_reranker=use_reranker,
                    use_hybrid=use_hybrid,
                )

            answer = result["answer"] or "검색 결과가 없거나 답변 생성에 실패했습니다."
            docs = result.get("docs") or []
            hwarn = result.get("hybrid_warning") or ""
            source = result.get("source", "")
            source_label = _SOURCE_LABELS.get(source, source)

            st.session_state.chat.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                if hwarn:
                    st.warning(hwarn)
                st.markdown(answer)
                if source_label:
                    st.caption(f"경로: {source_label}")
                if docs:
                    with st.expander(f"근거 청크 보기 (top {len(docs)})", expanded=False):
                        rank_label = "rerank" if use_reranker else "vector"
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
