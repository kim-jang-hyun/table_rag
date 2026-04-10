"""Streamlit UI for the Table RAG agent.

Run from the project root::

    streamlit run app/streamlit_app.py
"""

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

# Allow ``import rag_agent`` from the same directory when running via
# ``streamlit run app/streamlit_app.py`` (Streamlit adds the script directory
# to sys.path automatically, but we add it explicitly for safety).
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from dotenv import load_dotenv

import table_rag
import rag_agent
from table_rag.models import (
    get_qdrant_client,
    is_fastembed_available,
    load_embed_model,
    load_reranker,
    load_sparse_embedder,
)

# Project root — used to resolve relative folder paths from the sidebar.
_PROJECT_ROOT = Path(__file__).parent.parent

st.set_page_config(page_title="Table RAG 기반 검색 Agent", page_icon="📄", layout="wide")

_SOURCE_LABELS: Dict[str, str] = {
    "rag_generate": "📄 RAG 검색 (벡터 DB)",
    "web_fallback": "🌐 웹 검색 (문서 부족 시 폴백)",
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
    return load_embed_model()


@st.cache_resource
def _reranker_cached():
    return load_reranker()


@st.cache_resource
def _qdrant_cached():
    return get_qdrant_client()


_SUPPORTED_EXTENSIONS = (".pdf", ".pptx", ".ppt")


def _list_local_docs(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted([
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
    ])


def _ingest_pdfs(
    *,
    pdf_paths: Sequence[Path],
    collection: str,
    extract_table_chunks: bool,
    enable_hybrid: bool,
    merge_cross_page_tables: bool,
) -> int:
    embed_model = _embed_model_cached()
    return table_rag.ingest_pdfs_to_qdrant(
        pdf_paths=list(pdf_paths),
        collection=collection,
        embed_model=embed_model,
        extract_table_chunks=extract_table_chunks,
        enable_hybrid=enable_hybrid,
        merge_cross_page_tables=merge_cross_page_tables,
    )


def _upsert_doc(
    *,
    doc_path: Path,
    collection: str,
    extract_table_chunks: bool,
    enable_hybrid: bool,
    merge_cross_page_tables: bool,
) -> int:
    embed_model = _embed_model_cached()
    return table_rag.upsert_document_to_qdrant(
        doc_path=doc_path,
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


def main() -> None:
    _load_env()

    st.title("📄 Table RAG 기반 검색 Agent")
    st.caption(
        "모든 질문은 먼저 벡터 DB(RAG)에서 검색하고, 관련 문서가 부족할 때만 Tavily 웹검색으로 폴백합니다."
    )

    if "models_ready" not in st.session_state:
        st.session_state.models_ready = False

    if not st.session_state.models_ready:
        with st.spinner("모델 로딩 중... 잠시만 기다려 주세요. (bge-m3, reranker, BM25)"):
            _embed_model_cached()
            _reranker_cached()
            _qdrant_cached()
            if is_fastembed_available():
                load_sparse_embedder()
        st.session_state.models_ready = True
        st.rerun()

    _ensure_env_from_sidebar()

    with st.sidebar:
        st.subheader("RAG 설정")

        hybrid_ok = is_fastembed_available()
        if not hybrid_ok:
            st.caption(
                "참고: fastembed가 없어 BM25 하이브리드는 동작하지 않습니다. "
                "같은 venv에서 `pip install fastembed` 후 앱을 다시 실행하세요."
            )
        collection = st.text_input(
            "Qdrant collection",
            value=os.environ.get("QDRANT_COLLECTION", table_rag.DEFAULT_COLLECTION),
        )
        openai_model = st.text_input(
            "OpenAI model",
            value=os.environ.get("OPENAI_MODEL", table_rag.DEFAULT_OPENAI_MODEL),
        )
        qdrant_top_k = st.slider("Qdrant top_k", min_value=5, max_value=50, value=5, step=1)
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
            "0", "false", "no",
        }
        merge_cross_page_tables = st.toggle(
            "페이지 간 표 병합",
            value=_env_mcp,
            disabled=not extract_table_chunks,
            help="켜면 여러 페이지에 걸친 표를 하나의 청크로 합칩니다. 변경 후 재인덱싱하세요.",
        )

        st.subheader("문서")
        _pdf_folder_env = os.environ.get("PDF_FOLDER", "테스트문서")
        _pdf_folder_raw = Path(_pdf_folder_env)
        pdf_folder = (
            _pdf_folder_raw
            if _pdf_folder_raw.is_absolute()
            else (_PROJECT_ROOT / _pdf_folder_raw)
        ).resolve()

        docs = _list_local_docs(pdf_folder)
        default_pdf = os.environ.get("PDF_PATH", table_rag.DEFAULT_PDF)

        folder_paths: List[Path] = []
        if docs:
            names = [p.name for p in docs]
            selected_names = st.multiselect(
                "로컬 문서 선택 (복수 가능)",
                options=names,
                default=names,
                help="같은 collection에 선택한 모든 문서(PDF/PPT)를 함께 인덱싱합니다.",
            )
            folder_paths = [pdf_folder / n for n in selected_names]
        else:
            st.info(
                "현재 폴더에서 문서를 찾지 못했어요. "
                "아래 업로더를 사용하거나 PDF/PPT 파일을 프로젝트 폴더에 두세요."
            )
            if not Path(default_pdf).exists():
                st.caption(f"단일 경로 힌트: `PDF_PATH={default_pdf}` (파일이 없으면 업로드만 사용)")

        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = 0

        uploaded_list = st.file_uploader(
            "또는 문서 업로드 (PDF / PPT / PPTX, 복수 가능)",
            type=["pdf", "pptx", "ppt"],
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.uploader_key}",
            help="업로드한 파일은 테스트문서 폴더에 저장되고, VectorDB에 추가(기존 컬렉션 유지)됩니다.",
        )
        upload_paths: List[Path] = []
        if uploaded_list:
            pdf_folder.mkdir(parents=True, exist_ok=True)
            for up in uploaded_list:
                dest = pdf_folder / up.name
                dest.write_bytes(up.getbuffer())
                upload_paths.append(dest)
            st.success(f"업로드 완료: {len(upload_paths)}개 파일 → `{pdf_folder.name}/`")

        upsert_clicked = st.button(
            "업로드 파일 추가 인덱싱",
            disabled=not upload_paths,
            use_container_width=True,
            help="업로드한 파일만 VectorDB에 추가합니다. 기존 컬렉션은 유지됩니다.",
        )

        fallback: List[Path] = []
        if not folder_paths and not upload_paths:
            p = Path(default_pdf)
            if p.exists():
                fallback = [p]

        pdf_paths = _dedupe_paths(list(folder_paths) + fallback)

        ingest_clicked = st.button(
            "인덱싱(전체 재구성) 실행",
            type="primary",
            use_container_width=True,
            help="선택한 로컬 문서로 컬렉션을 초기화 후 재인덱싱합니다.",
        )

    if upsert_clicked:
        try:
            if not upload_paths:
                st.error("추가할 업로드 파일이 없습니다.")
            else:
                for doc_path in upload_paths:
                    spin = (
                        f"'{doc_path.name}' → 의미+키워드 임베딩 후 Qdrant 추가..."
                        if use_hybrid
                        else f"'{doc_path.name}' → 의미 임베딩 후 Qdrant 추가..."
                    )
                    with st.spinner(spin):
                        n_chunks = _upsert_doc(
                            doc_path=doc_path,
                            collection=collection,
                            extract_table_chunks=extract_table_chunks,
                            enable_hybrid=use_hybrid,
                            merge_cross_page_tables=(
                                merge_cross_page_tables and extract_table_chunks
                            ),
                        )
                    mode = "하이브리드 (의미+키워드)" if use_hybrid else "의미 벡터만"
                    st.success(
                        f"추가 완료 ({mode}). `{doc_path.name}` · 청크 {n_chunks:,}개 → collection: `{collection}`"
                    )
                st.session_state.uploader_key += 1
                st.rerun()
        except Exception as e:
            st.exception(e)

    if ingest_clicked:
        try:
            if not pdf_paths:
                st.error("인덱싱할 문서가 없습니다. 폴더에서 선택하거나 파일을 업로드하세요.")
            else:
                missing = [str(p) for p in pdf_paths if not p.exists()]
                if missing:
                    st.error("파일을 찾을 수 없습니다:\n" + "\n".join(missing))
                else:
                    spin = (
                        f"{len(pdf_paths)}개 문서 → 의미+키워드 임베딩 후 Qdrant 업서트..."
                        if use_hybrid
                        else f"{len(pdf_paths)}개 문서 → 의미 임베딩 후 Qdrant 업서트..."
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
                        f"인덱싱 완료 ({mode}). 문서 {len(pdf_paths)}개 · 청크 {n_chunks:,}개 · "
                        f"collection: `{collection}`\n\n{doc_list}"
                    )
        except Exception as e:
            st.exception(e)

    st.divider()

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    models_ready = st.session_state.get("models_ready", False)
    question = st.chat_input("질문을 입력하세요", disabled=not models_ready)

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
                                f"**[{i}]**{doc_bit}  page={d.get('page')}  "
                                f"chunk_id={d.get('chunk_id')}  "
                                f"type={d.get('source_type')}  "
                                f"{rank_label}={d.get('rerank_score', 0.0):.4f}"
                            )
                            st.code(d.get("text", ""), language="text")
        except Exception as e:
            st.exception(e)


if __name__ == "__main__":
    main()
