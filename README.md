## table_rag

Qdrant Cloud에 PDF를 적재하고, `bge-m3`로 임베딩한 뒤 검색 결과를 `bge-reranker-v2-m3`로 재정렬하는 최소 RAG 예제입니다.

### 준비물

- **PDF 파일**: `[POSCO홀딩스]임원ㆍ주요주주특정증권등소유상황보고서(2026.03.10).pdf` 를 이 폴더에 넣기
- **Qdrant Cloud**: Cluster URL / API Key

### 설치

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 환경변수 설정

- `.env.example` 을 `.env` 로 복사 후 값 채우기
- `QDRANT_API_KEY` 는 **코드에 하드코딩하지 말고** `.env` 로만 관리 권장

### 실행

```bash
python basic_rag.py
```

### 동작

- 1) PDF에서 본문/테이블 추출
- 2) 본문은 일반 청킹, 테이블은 행/열 구조를 보존한 텍스트로 변환
- 3) `BAAI/bge-m3` 로 청크 임베딩
- 4) Qdrant 컬렉션 생성 후 upsert (`source_type: text|table` payload 포함)
- 5) 질문 임베딩으로 Qdrant Top-K 검색
- 6) `BAAI/bge-reranker-v2-m3` 로 재정렬 후 상위 결과 출력

