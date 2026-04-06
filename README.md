# Table RAG

Qdrant Cloud에 PDF를 적재하고, `bge-m3`로 임베딩한 뒤 검색 결과를 `bge-reranker-v2-m3`로 재정렬하고 OpenAI LLM으로 최종 답변을 생성하는 RAG 시스템입니다.
LangGraph 기반 라우팅으로 PDF 관련 질문은 벡터 DB 검색(RAG), 일반·최신 질문은 Tavily 웹검색(Web RAG)으로 자동 분기합니다.

---

## 빠른 시작

### 준비물

- **PDF 파일**: `테스트문서/` 폴더에 PDF 넣기 (또는 Streamlit UI에서 업로드)
- **Qdrant Cloud**: Cluster URL / API Key
- **OpenAI API Key**
- **Tavily API Key**: [app.tavily.com](https://app.tavily.com) 에서 발급

### 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 환경변수 설정

`.env.example`을 `.env`로 복사한 뒤 값을 채웁니다.

```env
QDRANT_URL=https://...
QDRANT_API_KEY=...
QDRANT_COLLECTION=my_collection
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
PDF_FOLDER=테스트문서
```

### 실행

```bash
# Streamlit UI (권장)
streamlit run streamlit_app.py

# CLI (basic_rag.py 단독)
python basic_rag.py
```

### 동작 흐름 (Streamlit UI)

1. 사이드바에서 PDF 선택 후 **인덱싱 실행** 버튼 클릭
2. 채팅창에 질문 입력
3. LangGraph Router가 자동 분류:
   - **PDF 문서 관련** → Qdrant 벡터 검색 → 문서 품질 평가 → RAG 답변
   - **일반/최신 정보** → Tavily 웹검색 → Web RAG 답변
   - **문서 검색 실패** → Tavily 웹검색 → Web RAG 폴백 답변
4. 답변 하단에 경로 표시: `📄 RAG 검색` / `🌐 웹 검색 (직접)` / `🌐 웹 검색 (폴백)`

---

## 1. 개발 진행 현황

| 단계 | 내용 | 완료일 |
|------|------|--------|
| 1 | RAG 구축 | '26.03.26 |
| 2 | 테이블 영역과 비테이블 영역 구분 | '26.03.27 |
| 3 | LLM 연결 | '26.03.30 |
| 4 | 하이브리드 서치 추가 | '26.04.01 |
| 5 | 연속 페이지 표 병합 로직 추가 | '26.04.01 |
| 6 | LangChain 기반으로 전환 (LangSmith 트레이싱 포함) | '26.04.02 |
| 7 | 테이블 인식 못하는 경우 찾아서 개선 | '26.04.03 |
| 8 | LLM까지 확장하자.(gpt-4o-mini가 qwen3-b32와 유사한 스펙) 평가agent만들어서 langgraph를 적용해 본다던지, 과제를 확장해보자.  | '26.04.03 |
| 9 | LangGraph 기반 라우팅 에이전트 도입 (Router → RAG / Web RAG 자동 분기, Tavily 웹검색 통합) | '26.04.06 |

---

## 2. 기술 스택

### 2.1 RAG 구성 요소

| 구성 요소 | 사용 기술 | 설명 |
|-----------|-----------|------|
| 에이전트 프레임워크 | LangGraph | StateGraph 기반 조건부 라우팅 에이전트 |
| 프레임워크 | LangChain | LCEL 체인, 리트리버, 리랭커 통합 |
| Vector Store | Qdrant Cloud | 고성능 벡터 검색, 하이브리드(dense+sparse) 지원 |
| Embedding | BAAI/bge-m3 | HuggingFaceEmbeddings 래퍼, cosine 정규화 |
| Sparse Embedding | Qdrant/bm25 | FastEmbedSparse (fastembed), 하이브리드 검색용 |
| Rerank | BAAI/bge-reranker-v2-m3 | HuggingFaceCrossEncoder + CrossEncoderReranker |
| LLM | gpt-4o-mini | ChatOpenAI (Qwen3-32B와 유사한 성능) |
| 웹검색 | Tavily | 실시간 웹검색, 최대 5개 결과 반환 |
| PDF 파싱 | PyMuPDF (fitz) | 텍스트 + 테이블 구조화 추출 (커스텀 로직) |
| UI | Streamlit | 멀티 PDF 업로드, 채팅 인터페이스 |
| 관찰성 | LangSmith | LANGSMITH_TRACING=true 설정 시 자동 트레이싱 |

### 2.2 테이블 영역 분리 이유

- **구조 보존**: 테이블은 행(Row)과 열(Column)의 관계가 핵심입니다. 일반 텍스트 파싱은 이 관계를 파괴하여 수치 왜곡을 초래합니다.
- **정확한 검색**: 테이블 제목(Caption)과 내부 수치 데이터를 하나의 문맥(Context)으로 인덱싱하여 검색 품질을 높입니다.

**데이터 샘플 (JSON)**

```json
{
  "doc": "POSCO홀딩스_주식 소각 결정_(2026.02.19).pdf",
  "page": 1,
  "source_type": "table",
  "text": "TABLE: 주식 소각 결정\n소각할 주식의 종류: 보통주식\n소각할 주식의 수: 1,691,425 (주)\n소각예정금액: 635,130,087,500 (원)"
}
```

### 2.3 하이브리드 서치

벡터 검색과 BM25 키워드 검색을 결합하여 **정확도(Precision)**와 **재현율(Recall)**을 동시에 개선합니다.

| 방식 | 설명 |
|------|------|
| 벡터 검색 (Vector Search) | 의미 기반 검색 → 문장 임베딩으로 유사 결과 반환 |
| BM25 검색 (Keyword Search) | 단어 기반 검색 → 키워드가 정확히 일치하는 결과 우선 반환 |

---

## 3. RAG 파이프라인 및 핵심 구현

### 3.1 전체 파이프라인 흐름

```mermaid
flowchart TD
    PDF["PDF (PyMuPDF)"]
    Text["텍스트 청크\nsize=1000 / overlap=150"]
    Table["테이블 청크\nTABLE 구조화 텍스트"]
    Merge["교차 페이지\n표 병합 로직"]
    LCDocs["LangChain Documents\npage_content + metadata"]
    Embed["HuggingFaceEmbeddings\nBAAI/bge-m3"]
    Sparse["FastEmbedSparse\nQdrant/bm25"]
    QV["QdrantVectorStore\nHYBRID / DENSE"]
    Query["사용자 질문"]
    Retriever["as_retriever\ntop-k=20"]
    Reranker["ContextualCompressionRetriever\nbge-reranker-v2-m3 top-n=5"]
    Chain["LCEL Chain\nprompt or ChatOpenAI or StrOutputParser"]
    LS["LangSmith\n자동 트레이싱"]
    Answer["한국어 답변"]

    PDF --> Text
    PDF --> Table
    Table --> Merge --> LCDocs
    Text --> LCDocs
    LCDocs --> Embed --> QV
    LCDocs --> Sparse --> QV
    Query --> Retriever --> Reranker --> Chain --> Answer
    QV --> Retriever
    Chain -. trace .-> LS
```

### 3.2 핵심 구현 아이디어

| 아이디어 | 구현 방법 | 효과 |
|----------|-----------|------|
| 텍스트 청킹 | 1000자 슬라이딩 윈도우, overlap 150자 | 문맥 단절 최소화 |
| 테이블 구조화 | `page.find_tables()` → `TABLE / columns / rowN` 텍스트 포맷 | 행/열 관계 보존 |
| 교차 페이지 표 병합 | 연속 페이지 + 동일 열 수 + bbox 위치 휴리스틱(하단 30% / 상단 72%) | 분할된 표를 하나의 청크로 처리 |
| 하이브리드 검색 | `RetrievalMode.HYBRID` (dense + BM25 sparse, RRF 융합) | 정확도·재현율 동시 개선 |
| 리랭킹 | `CrossEncoderReranker` + `ContextualCompressionRetriever` | top-20 → top-5 정밀 선별 |
| LCEL 체인 | `ChatPromptTemplate \| ChatOpenAI \| StrOutputParser` | 구성 가능한 파이프라인 |
| LangSmith 트레이싱 | `.env`의 `LANGSMITH_TRACING=true`만으로 모든 체인 자동 추적 | 디버깅·성능 분석 |

### 3.3 청크 ID 포맷 및 메타데이터

**청크 ID 규칙**

```
텍스트 청크:  {doc_slug}::p{page:04d}_t{j:03d}
테이블 청크:  {doc_slug}::p{page:04d}_tb{k:03d}

예시)
  posco_report::p0002_t001   ← 2페이지 첫 번째 텍스트 청크
  posco_report::p0003_tb001  ← 3페이지 첫 번째 테이블 청크
```

**Qdrant 페이로드 메타데이터**

| 필드 | 설명 | 예시 |
|------|------|------|
| doc | 원본 PDF 파일명 | POSCO홀딩스_주식소각.pdf |
| page | 페이지 번호 | 2 |
| chunk_id | 청크 고유 ID | posco::p0002_tb001 |
| source_type | 청크 유형 | text / table |

### 3.4 LangChain 전환 주요 변경점 ('26.04.02)

| 기존 (직접 구현) | 변경 후 (LangChain) |
|-----------------|---------------------|
| `SentenceTransformer` + 수동 `.encode()` | `HuggingFaceEmbeddings` |
| `CrossEncoder` + 수동 `.predict()` | `HuggingFaceCrossEncoder` + `CrossEncoderReranker` |
| `openai.OpenAI` + `responses.create()` | `ChatOpenAI` + LCEL 체인 |
| Qdrant `PointStruct` 수동 upsert (~70줄) | `QdrantVectorStore.from_documents()` |
| 수동 `query_points` + RRF 로직 (~100줄) | `ContextualCompressionRetriever` |

### 3.5 테이블 추출 및 교차 페이지 병합 로직 상세

#### 테이블 분리 로직

**Step 1 — 테이블 감지 (`_extract_raw_tables_from_doc`)**

PyMuPDF의 `page.find_tables()`로 각 페이지에서 테이블 영역을 자동 인식합니다.
PDF의 선(line), 배경색, 텍스트 정렬 패턴을 분석해 테이블을 감지하고, 각 테이블마다 `bbox`(좌표), `rows`(행 데이터), `page_height`를 저장합니다.

```python
finder = page.find_tables()        # PyMuPDF 내장 테이블 감지
raw_rows = table.extract()         # 행/열 데이터 추출
bbox = tuple(float(x) for x in table.bbox)
```

**Step 2 — 테이블 제목 감지 (`_find_table_title`)**

테이블 위 가장 가까운 텍스트 블록을 제목으로 추출하는 휴리스틱입니다.

```python
# 조건: 테이블 상단 y좌표보다 위에 있고, 120자 이하인 텍스트
if float(y1) <= table_top + 2.0 and len(content) <= 120:
    candidates.append((distance, content))
candidates.sort(key=lambda x: x[0])  # 가장 가까운 것 선택
```

**Step 3 — 구조화 텍스트 변환 (`_table_to_text`)**

테이블을 LLM이 이해하기 좋은 텍스트 포맷으로 직렬화합니다. 빈 셀은 건너뛰고 `열이름=값` 형식으로 행/열 관계를 보존합니다.

```
TABLE
title: 주식 소각 결정
columns: 종류, 주식수, 금액
row1: 종류=보통주식 | 주식수=1,691,425 | 금액=635,130,087,500
```

---

#### 교차 페이지 표 병합 로직 (`_merge_cross_page_raw_tables`)

다음 **5가지 조건을 모두 만족할 때만** 두 테이블을 하나로 병합합니다.

| 조건 | 판별 방법 |
|------|-----------|
| 연속 페이지 | `nxt["page"] == cur["end_page"] + 1` |
| 이전 페이지 마지막 표 | 같은 페이지에 뒤따르는 표가 없음 |
| 다음 페이지 첫 번째 표 | 같은 페이지에 앞서는 표가 없음 |
| 열 수 동일 | `len(rows_a[0]) == len(rows_b[0])` |
| 위치 휴리스틱 | 이전 표 끝 ≥ 페이지 높이의 30% + 다음 표 시작 ≤ 페이지 높이의 72% |

**위치 휴리스틱 (`_merge_geometry_suggests_split`)**

```python
(py1 / ph_p >= 0.30) and (ny0 / ph_n <= 0.72)
# py1/ph_p: 이전 페이지에서 표 끝 위치 비율 (중간 이하 = 하단에 표 있음)
# ny0/ph_n: 다음 페이지에서 표 시작 위치 비율 (상단 72% 이내 = 상단 근처)
```

**반복 헤더 제거 (`_continuation_body_rows`)**

PDF에서 표가 페이지를 넘어갈 때 헤더가 반복되는 경우, 자동으로 감지해서 제거합니다.

```python
if _row_cells_equal(header_row, next_page_rows[0]):
    return next_page_rows[1:]  # 다음 페이지 첫 행이 헤더와 동일하면 제거
```

---

## 4. LangGraph 라우팅 에이전트 (`rag_agent.py`)

### 4.1 도입 배경

기존 시스템은 항상 벡터 DB 검색을 수행했습니다. 그러나 질문 유형에 따라 최적의 답변 경로가 다릅니다.

| 질문 유형 | 예시 | 최적 경로 |
|-----------|------|-----------|
| 문서 관련 (RAG 주제) | "포스코홀딩스 주식 소각 수량은?" | 벡터 DB 검색 → LLM |
| 일반/최신 정보 (일반 주제) | "2026년 AI 트렌드는?" | 웹검색(Tavily) → LLM |
| 문서 검색했지만 관련 내용 없음 | 검색 문서가 질문과 무관할 때 | 웹검색(Tavily) → LLM (폴백) |

LangGraph를 도입해 **질문 유형을 자동 분류**하고, 세 가지 경로 중 가장 적합한 경로로 자동 분기합니다.

---

### 4.2 전체 아키텍처

```mermaid
flowchart TD
    UserQ([사용자 질문]) --> RouterNode["router_node\nRAG 주제인지 분류"]

    RouterNode -->|"RAG 주제\n문서 관련 질문"| RetrieveNode["retrieve_node\n벡터 DB 검색\nQdrant + 리랭커"]
    RouterNode -->|"일반 주제\n웹 검색 필요"| WebDirectNode["web_rag_node\nTavily 웹검색\n+ LLM 직접호출"]

    RetrieveNode --> GradeNode["grade_node\n문서 품질 평가\n충분 / 부족"]

    GradeNode -->|"충분\n관련 문서 있음"| GenerateNode["generate_node\nRAG 기반 답변\n벡터 DB 컨텍스트 사용"]
    GradeNode -->|"문서 부족\n관련 내용 없음"| WebFallbackNode["web_rag_node\nTavily 웹검색\n+ LLM 폴백"]

    GenerateNode --> FinalAns([최종 응답\nsource: rag_generate])
    WebDirectNode --> FinalAns2([최종 응답\nsource: web_direct])
    WebFallbackNode --> FinalAns3([최종 응답\nsource: web_fallback])
```

---

### 4.3 AgentState (그래프 공유 상태)

LangGraph의 모든 노드는 `AgentState`를 읽고 업데이트합니다. 한 노드가 반환한 값은 다음 노드에서 그대로 참조됩니다.

```python
class AgentState(TypedDict):
    # 입력
    question: str           # 사용자 질문

    # 라우팅 결과
    route: str              # "rag" | "general"

    # RAG 경로
    documents: list         # Qdrant 검색 결과 dict 리스트
    grade: str              # "sufficient" | "insufficient"

    # 최종 출력
    answer: str             # LLM 생성 답변
    source: str             # "rag_generate" | "web_direct" | "web_fallback"
    hybrid_warning: str     # 하이브리드 검색 경고 메시지 (해당 시)

    # 검색 설정 (Streamlit에서 주입)
    collection: str
    use_reranker: bool
    use_hybrid: bool
    qdrant_top_k: int
    rerank_top_k: int
    openai_model: str
```

---

### 4.4 노드별 상세 설명

#### `router_node` — RAG 주제 vs 일반 주제 분류

**역할**: 사용자 질문을 보고 벡터 DB 검색이 필요한 질문인지(`"rag"`), 웹검색이 더 적합한 질문인지(`"general"`)를 LLM이 판단합니다.

**분류 기준 (LLM 프롬프트)**

```
- "rag"    : 업로드된 PDF 문서(공시, 보고서, 계약서, 재무제표 등 특정 문서)의 내용을 묻는 질문
- "general": 일반 상식, 최신 뉴스, 인터넷 검색이 필요한 질문,
             또는 특정 문서 없이 답할 수 있는 질문
```

**실제 분류 예시**

| 질문 | 분류 | 이유 |
|------|------|------|
| "포스코홀딩스 주식 소각 수량은?" | `rag` | 특정 공시 문서 내용 질문 |
| "주주총회 의안 내용은?" | `rag` | 업로드된 PDF 내 정보 |
| "2026년 글로벌 철강 시장 전망은?" | `general` | 최신 시장 정보, 웹검색 필요 |
| "파이썬 리스트와 튜플의 차이는?" | `general` | 일반 상식, 문서 무관 |
| "포스코 현재 주가는 얼마야?" | `general` | 실시간 정보, 웹검색 필요 |

**출력 형식**: LLM이 `JsonOutputParser`를 통해 `{"route": "rag"}` 또는 `{"route": "general"}` JSON만 반환.
파싱 실패 또는 예상 외 값이 오면 `"general"`로 안전하게 fallback합니다.

```python
chain = route_prompt | llm | JsonOutputParser()
result = chain.invoke({"question": user_input})
route = result.get("route", "general")
if route not in ("rag", "general"):
    route = "general"   # 안전 fallback
```

---

#### `retrieve_node` — 벡터 DB 검색

**역할**: `route == "rag"`일 때 실행. 기존 `basic_rag.search_and_rerank()`를 호출하여 Qdrant에서 관련 청크를 검색합니다.

- 하이브리드 검색(dense + BM25 sparse, RRF 융합) 또는 dense 단독 검색
- BGE-reranker로 top-20 → top-5 정밀 선별 (설정에 따라)
- 결과를 `documents` 필드에 저장하여 다음 노드로 전달

---

#### `grade_node` — 문서 품질 평가

**역할**: 검색된 문서들이 질문에 답하기에 충분한지 LLM이 평가합니다.

**판단 기준 (LLM 프롬프트)**

```
- "sufficient"  : 검색된 문서 중 질문과 관련된 내용이 있어 답변 생성이 가능한 경우
- "insufficient": 검색된 문서가 없거나, 질문과 관련 없는 내용만 있어 답변 불가한 경우
```

**평가 방식**: 검색된 문서 상위 3개의 앞 300자를 발췌해 LLM에게 "이 문서들로 질문에 답할 수 있나?"를 물어봅니다.

```python
context_preview = "\n".join(d.get("text", "")[:300] for d in docs[:3])
# LLM이 {"grade": "sufficient"} 또는 {"grade": "insufficient"} 반환
```

**문서가 0개**이면 LLM 호출 없이 바로 `"insufficient"`로 처리합니다.

---

#### `generate_node` — RAG 기반 답변 생성

**역할**: `grade == "sufficient"`일 때 실행. 기존 `basic_rag.answer_with_openai()`를 호출하여 검색된 문서를 컨텍스트로 LLM 답변을 생성합니다.

- 프롬프트: "반드시 제공된 컨텍스트 범위 안에서만 답하세요. 근거가 부족하면 모른다고 말하세요."
- 답변 끝에 근거 청크 번호(`[1]`, `[2]`...) 표시
- `source = "rag_generate"` 설정

---

#### `web_rag_node` — Tavily 웹검색 + LLM (직접호출 & 폴백 공용)

**역할**: `route == "general"` (직접 호출) 또는 `grade == "insufficient"` (폴백) 두 경우 모두 이 노드를 사용합니다. `source` 필드 값으로 어느 경로에서 왔는지 구분합니다.

**처리 흐름**

```
1. Tavily API 호출 → 웹 검색 결과 최대 5개 수집
2. 각 결과의 URL + 내용을 컨텍스트 문자열로 조합
3. "웹 검색 결과를 근거로 답하되, 없는 내용은 추측하지 마세요" 프롬프트 + LLM 호출
4. 답변 끝에 참고 URL 표시
```

**source 설정 로직**

```python
# route가 "rag"인데 여기까지 왔다면 → 폴백 (grade: insufficient)
# route가 "general"이면 → 직접 호출
source = "web_fallback" if state.get("route") == "rag" else "web_direct"
```

**Tavily 설정**

```python
tool = TavilySearchResults(max_results=5, tavily_api_key=os.environ["TAVILY_API_KEY"])
results = tool.invoke(state["question"])
# results: [{"url": "https://...", "content": "검색 결과 본문"}, ...]
```

---

### 4.5 조건부 엣지 (라우팅 로직 요약)

```
START
  └─► router_node
        ├─ route == "rag"      ──► retrieve_node
        │                              └─► grade_node
        │                                    ├─ grade == "sufficient"   ──► generate_node ──► END
        │                                    └─ grade == "insufficient" ──► web_rag_node  ──► END
        └─ route == "general"  ──────────────────────────────────────────► web_rag_node  ──► END
```

---

### 4.6 경로별 응답 레이블 (Streamlit UI)

Streamlit 화면에서는 `source` 값에 따라 답변 아래에 어느 경로로 응답이 생성됐는지 표시합니다.

| source 값 | UI 표시 | 의미 |
|-----------|---------|------|
| `rag_generate` | 📄 RAG 검색 (벡터 DB) | Qdrant 검색 + 문서 기반 답변 |
| `web_direct` | 🌐 웹 검색 (직접 호출) | Router가 일반 주제로 분류 → Tavily |
| `web_fallback` | 🌐 웹 검색 (폴백 — 문서 부족) | Qdrant 검색했지만 관련 문서 없음 → Tavily |

---

## 5. 정확도 테스트 (Golden Dataset)

### A. 공시 데이터 정밀 추출
> 문서: `POSCO홀딩스_주식 소각 결정_(2026.02.19)주식 소각 결정.pdf`

| ID | 질문 | 정답 |
|----|------|------|
| Q1 | 이번 공시에서 소각하기로 결정한 '보통주식'의 총 수량은 몇 주인가? | **1,691,425주** |
| Q2 | '소각예정금액'은 총 얼마인가? (원 단위 전체 기입) | **635,130,087,500원** |
| Q3 | 주식 소각을 결정한 '이사회결의일'은 언제인가? | **2026-02-19** |
| Q4 | 발행주식 총수 대비 이번에 소각하는 주식의 비율은 약 몇 %인가? (소수점 둘째 자리 반올림) | **약 2.09%** |
| Q5 | 소각예정금액과 소각 주식 수를 바탕으로 역산한 '1주당 소각 평균 단가'는 얼마인가? | **약 375,500원** |
| Q6 | "기타 투자판단과 관련한 중요사항"에 따르면, 이번 주식 소각은 어떤 이익을 재원으로 하는가? | **배당가능이익 범위 내에서 취득한 자기주식** |
| Q7 | 이번 주식 소각 이후 자본금의 감소가 발생하는가? | **아니오 (자본금 감소 없음)** |
| Q8 | "기타 투자판단과 관련한 중요사항"의 내용을 항목별로 요약하여 알려줘. | **재원(배당가능이익), 법적근거(상법 제343조), 수량(발행주식의 2%), 금액(이사회 전일 종가) 등 요약** |

### B. 병합 테이블 구조 해석
> 문서: `병합된 테이블 테스트 샘플.pdf`

| ID | 질문 | 정답 |
|----|------|------|
| Q9 | 에너지 솔루션 부문의 2026년 영업이익은 얼마인가? | **124,000** |
| Q10 | 지능형 로보틱스 부문의 비고란 내용은 무엇인가? | **물류 자동화 로봇 수주 증가 및 R&D 비용 효율화** |
| Q11 | 전사 합계 매출 성장률에 가장 높게 기여한 사업부문은 어디인가? | **지능형 로보틱스 (+97.8%)** |
| Q12 | 에너지 솔루션의 2025년 대비 2026년 매출 증감액은 얼마인가? | **330,000** |

### C. 페이지 초월 및 데이터 정정
> 문서: `POSCO홀딩스_주주총회소집결의_(2026.02.19)주주총회소집결의.pdf`

| ID | 질문 | 정답 |
|----|------|------|
| Q13 | 김주연 사외이사 후보의 'P&G 한국/일본지역 부회장' 재임 기간은 언제인가? | **2019~2022년** |
| Q13-1 | [사외이사선임 세부내역]에서 김주연씨의 'P&G 한국 대표이사 사장' 재임 기간은? | **2016~2018년** |
| Q14 | 정정 공시의 '4. 정정사항'에서 제2-2호 의안의 '정정 후' 내용은 무엇인가? | **분리선출 감사위원 수 증원** |
| Q15 | 김준기 후보가 현재 '포스코홀딩스 사외이사'로 재직하기 시작한 연도는? | **2023년** |
| Q16 | 이주태 후보의 임기 및 신규선임 여부는 무엇인가? | **임기 1년, 재선임** |

### D. 다중 컬럼 헤더 · 수식 헤더 · 합계 행 해석
> 문서: `[POSCO홀딩스]임원ㆍ주요주주특정증권등소유상황보고서(2026.03.10).pdf`

> **테스트 포인트**: A~H 8개 컬럼 레이블 매핑, 수식이 포함된 헤더 셀, 시점별 행 비교, 세부변동내역 다중 행 및 합계 행 읽기

| ID | 질문 | 정답 |
|----|------|------|
| Q17 | '나. 특정증권등의 종류별 소유내역' 표에서 이익참가부사채권에 해당하는 컬럼 알파벳은? | **E** |
| Q18 | 같은 표에서 전환사채권(C), 신주인수권부사채권(D), 이익참가부사채권(E)의 소유 수량은 각각 얼마인가? | **모두 '-' (보유 없음)** |
| Q19 | '가. 소유 특정증권등의 수 및 소유비율' 표에서 직전보고서 대비 이번보고서 기준 증가한 주권 수량은? | **31주** (335 → 366) |
| Q20 | 발행주식 총수(J) 계산 표 헤더에 명시된 '특정증권등의 소유비율' 계산식 전체를 그대로 써라. | **[A+I / J+I-(F+G+H)] × 100** |
| Q21 | 세부변동내역 표에서 2026.01.30 장내매수와 2026.03.05 장내매수의 취득단가를 각각 써라. | **378,000원 / 374,500원** |
| Q22 | 세부변동내역 합계 행에서 변동전·증감·변동후 주식 수를 각각 써라. | **335주 / 31주 / 366주** |
| Q23 | 세부변동내역 합계 행에 표시된 평균 취득단가는 얼마인가? | **376,419원** |
| Q24 | '나. 특정증권등의 종류별 소유내역' 표에서 알파벳 H열의 증권 종류명과 소유 수량은? | **기타, '-' (보유 없음)** |

### E. 2단 열 헤더 (Multi-level Column Headers)
> 문서: `복잡한_테이블_테스트_샘플.pdf`

> **테스트 포인트**: 상위 헤더(매출/비용/이익)와 하위 헤더(국내/해외/합계 등)의 2단 계층 매핑 — 특정 셀 값을 읽으려면 두 단계의 헤더를 동시에 참조해야 함. '부문' 열은 rowspan 2로 병합됨.

| ID | 질문 | 정답 |
|----|------|------|
| Q25 | '반도체' 부문의 '해외 매출'은 얼마인가? (단위: 백만원) | **132,000** |
| Q26 | '소재' 부문의 '비용 합계'는 얼마인가? (단위: 백만원) | **46,000** |
| Q27 | 세 부문(반도체·디스플레이·소재) 중 '순이익'이 가장 높은 부문은? | **반도체 (71,200백만원)** |
| Q28 | '디스플레이' 부문의 영업이익률은? (영업이익 ÷ 매출합계, 소수점 첫째 자리 반올림) | **약 36.1%** (34,500 ÷ 95,500) |

### F. 행/열 병합 혼재 (Mixed Rowspan + Colspan)
> 문서: `복잡한_테이블_테스트_샘플.pdf`

> **테스트 포인트**: 제품군(A계열/B계열) rowspan 2 + 분기 헤더(1분기/2분기) colspan 3 + 합계 행 colspan 2가 동시에 존재하는 복합 병합 구조. 병합된 제품군 셀에서 하위 제품(A-1, A-2)의 값을 올바르게 식별해야 함.

| ID | 질문 | 정답 |
|----|------|------|
| Q29 | 'A-2' 제품의 2분기 달성률은 얼마인가? | **110.0%** |
| Q30 | 'B계열' 전체(B-1 + B-2)의 1분기 실적 합계는 얼마인가? | **350** (200 + 150) |
| Q31 | 1분기와 2분기를 통틀어 달성률이 가장 높은 제품과 그 수치는? | **A-1, 1분기 120.0%** |

### G. 카테고리 행 + 소계 테이블 (Category Rows + Subtotals)
> 문서: `복잡한_테이블_테스트_샘플.pdf`

> **테스트 포인트**: 【직접비】·【간접비】 카테고리 행(colspan 3)이 그룹 구분자 역할을 하고, 각 그룹 하단에 소계 행, 최하단에 총합 행이 있는 3단 계층 구조. 카테고리 행을 데이터 행으로 오인하거나 소계와 총합을 혼동하는 경우를 검증.

| ID | 질문 | 정답 |
|----|------|------|
| Q32 | '직접비' 소계는 얼마인가? (단위: 백만원) | **51,400** |
| Q33 | '간접비' 항목 중 금액이 가장 큰 항목은 무엇이며 얼마인가? | **판관비 (41,500백만원)** |
| Q34 | '외주비'가 총 비용에서 차지하는 비중은? | **5.6%** |
| Q35 | 직접비와 간접비를 모두 합산한 총 합계는 얼마인가? (단위: 백만원) | **156,000** |

### H. 양방향 헤더 매트릭스 (Bidirectional Header Matrix)
> 문서: `복잡한_테이블_테스트_샘플.pdf`

> **테스트 포인트**: 행 헤더(지역)와 열 헤더(제품)가 모두 존재하는 크로스탭 구조. 특정 값을 읽으려면 행·열 헤더를 동시에 참조해야 하며, 합계 행/열이 마지막에 위치함.

| ID | 질문 | 정답 |
|----|------|------|
| Q36 | '부산' 지역에서 'B제품'의 판매량은 얼마인가? | **1,200** |
| Q37 | '서울' 지역의 전 제품 판매량 합계는 얼마인가? | **4,690** |
| Q38 | 전체 지역 중 'C제품' 판매량이 가장 많은 지역은 어디이며, 판매량은? | **서울, 2,100개** |
| Q39 | 네 제품(A·B·C·D) 중 전국 판매량 합계가 가장 많은 제품은? | **C제품 (4,350개)** |

---

## 6. 정확도 테스트 결과 기록표

| 질문 ID | 질문 유형 | 정답 (Ground Truth) | Basic RAG | Table-Separated RAG | 비고 |
|---------|-----------|---------------------|-----------|---------------------|------|
| Q1 | 수치 추출 | 1,691,425주 | - | - | 보통주 수량 |
| Q2 | 금액 인식 | 635,130,087,500원 | - | - | 소각예정금액 |
| Q3 | 날짜 매칭 | 2026-02-19 | - | - | 이사회결의일 |
| Q4 | 비중 계산 | 약 2.09% | - | - | 산술 능력 |
| Q5 | 단가 추론 | 약 375,500원 | - | - | 산술 능력 |
| Q6 | 텍스트 추출 | 배당가능이익 | - | - | 재원 확인 |
| Q7 | 맥락 추론 | 자본금 감소 없음 | - | - | 재무 영향 |
| Q8 | 요약/추출 | 항목별 상세 요약 | - | - | 복합 텍스트 |
| Q9 | 세로 병합 | 124,000 | - | - | Rowspan 해석 |
| Q10 | 가로 병합 | 물류 자동화 로봇... | - | - | Colspan 해석 |
| Q11 | 복합 추론 | 지능형 로보틱스 | - | - | 데이터 비교 |
| Q12 | 산술 연산 | 330,000 | - | - | 증감액 계산 |
| Q13 | 페이지 초월 | 2019~2022년 | - | - | 표 분절 테스트 |
| Q13-1 | 페이지 초월 | 2016~2018년 | - | - | 표 분절 테스트 |
| Q14 | 정정 구분 | 분리선출 감사위원 수 증원 | - | - | 컬럼 매칭 |
| Q15 | 정보 통합 | 2023년 | - | - | 멀티 청크 통합 |
| Q16 | 테이블 추출 | 1년, 재선임 | - | - | 행/열 관계 |
| Q17 | 컬럼 레이블 매핑 | E | - | - | A~H 헤더 파싱 |
| Q18 | 다중 컬럼 값 읽기 | 모두 '-' (보유 없음) | - | - | 8열 헤더 대응 |
| Q19 | 행 간 비교 | 31주 | - | - | 시점별 행 비교 |
| Q20 | 수식 헤더 추출 | [A+I / J+I-(F+G+H)] × 100 | - | - | 수식 포함 헤더 |
| Q21 | 다중 행 읽기 | 378,000원 / 374,500원 | - | - | 날짜별 행 대응 |
| Q22 | 합계 행 추출 | 335주 / 31주 / 366주 | - | - | 합계 행 파싱 |
| Q23 | 가중평균 읽기 | 376,419원 | - | - | 합계 행 단가 |
| Q24 | 열 레이블 연결 | 기타, '-' | - | - | 컬럼명 역매핑 |
| Q25 | 2단헤더 — 특정 셀 읽기 | 132,000백만원 | - | - | 해외매출(반도체) |
| Q26 | 2단헤더 — 특정 셀 읽기 | 46,000백만원 | - | - | 비용합계(소재) |
| Q27 | 2단헤더 — 열 간 비교 | 반도체 (71,200) | - | - | 최대 순이익 부문 |
| Q28 | 2단헤더 — 계산 | 약 36.1% | - | - | 영업이익률 |
| Q29 | 혼재병합 — 하위행 읽기 | 110.0% | - | - | A-2 2분기 달성률 |
| Q30 | 혼재병합 — rowspan 합산 | 350 | - | - | B계열 1분기 실적 합 |
| Q31 | 혼재병합 — 전체 비교 | A-1 1분기 120.0% | - | - | 최고 달성률 |
| Q32 | 카테고리+소계 — 소계 읽기 | 51,400백만원 | - | - | 직접비 소계 |
| Q33 | 카테고리+소계 — 항목 비교 | 판관비 (41,500) | - | - | 간접비 최대 항목 |
| Q34 | 카테고리+소계 — 비중 읽기 | 5.6% | - | - | 외주비 비중 |
| Q35 | 카테고리+소계 — 총합 읽기 | 156,000백만원 | - | - | 총 합계 |
| Q36 | 매트릭스 — 교차점 읽기 | 1,200 | - | - | 부산×B제품 |
| Q37 | 매트릭스 — 행 합계 | 4,690 | - | - | 서울 합계 |
| Q38 | 매트릭스 — 열 간 비교 | 서울, 2,100 | - | - | C제품 최대 지역 |
| Q39 | 매트릭스 — 열 합계 비교 | C제품 (4,350) | - | - | 최다 판매 제품 |
| **합계** | | **성공률(%)** | **(미정)** | **(미정)** | **39개 질문 기준** |

---

## 7. 참고

> **No RAG 제외 사유**: LLM 직접 질의 시 컨텍스트 및 토큰 한계로 인해 모든 문서를 첨부하여 테스트하는 것은 현실적으로 불가능함.
> LLM보다는 RAG에 집중할 것!!! (Vectore DB에 Table이 제대로 들어갔는지 확인 필수)
