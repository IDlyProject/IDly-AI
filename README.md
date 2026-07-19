# IDly-AI-test

메일 박스(`.mbox`) 파일을 업로드해 보안 관련 메일을 분석하고, 발신자 단위로 위험도와 해석을 반환하는 FastAPI 기반 프로젝트입니다.

## 프로젝트 개요

이 서비스는 메일 제목과 본문에서 키워드를 찾고, 발신자별 패턴을 가공해 계정 단위 요약 결과를 생성합니다. 업로드된 파일은 분석 후 즉시 삭제되며, 건강 체크 엔드포인트도 함께 제공합니다.

## 개발 환경

- 운영체제: Windows 기준으로 작업 가능
- 실행 언어: Python
- 웹 프레임워크: FastAPI
- ASGI 서버: Uvicorn
- 저장소: SQLite(작업 큐/상태 관리용)
- 입력 형식: `.mbox`

## 스택 정보

- `FastAPI`: API 서버와 요청/응답 모델 처리
- `Uvicorn`: 로컬 개발 및 배포용 ASGI 실행
- `SQLite` + `SQLAlchemy`: 작업 큐와 상태 저장
- `Pydantic`: 응답 스키마 검증
- `httpx`: 비동기 HTTP 요청, 배포 환경의 keep-alive 용도
- `pandas`, `numpy`: 데이터 전처리와 집계
- `scikit-learn`: TF-IDF, KMeans, Isolation Forest, SVD 기반 분석
- `hdbscan`(선택): 설치되어 있으면 추가 군집 분석에 사용

## 주요 기능

- `.mbox` 파일 업로드 및 분석
- 제목/본문에서 키워드 매칭
- 발신자별 보안 점수 산정
- 문제 메일 목록 반환
- `GET /health` 헬스 체크

## 설치

가상환경을 만든 뒤 의존성을 설치합니다.

```bash
pip install -r requirements.txt
```

## 실행

로컬 개발 서버를 실행합니다.

```bash
uvicorn app.main:app --reload
```

기본적으로 `http://127.0.0.1:8000` 에서 확인할 수 있습니다.

## API

### `GET /health`

서버 상태를 확인합니다.

응답 예시:

```json
{ "status": "ok" }
```

### `POST /analyze`

`multipart/form-data`로 `.mbox` 파일을 업로드하면 분석 결과를 반환합니다.

요청 예시:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
	-F "file=@mock_security.mbox"
```

응답은 계정 목록을 포함하며, 각 계정에는 보안 점수, 보안 레벨, 해석, 문제 메일 목록이 포함됩니다.

## 동작 메모

- 업로드 파일은 `uploads/`에 임시 저장된 뒤 분석이 끝나면 삭제됩니다.
- 기본 분석 키워드는 `보안`입니다.
- `RENDER_EXTERNAL_URL` 환경 변수가 설정되면 배포 환경에서 `/health`로 주기적인 keep-alive 요청을 보냅니다.
- SQLite는 `app/db.py`에서 `jobs.db`로 설정되며, `app/models.py`, `app/service.py`, `app/worker.py`의 작업 큐/상태 관리에 사용됩니다. 현재 `/analyze` 엔드포인트 자체는 SQLite를 거치지 않습니다.

## 프로젝트 구조

```text
analysis_pipeline.py
app/
	db.py
	main.py
	models.py
	schemas.py
	service.py
	worker.py
uploads/
mock.mbox
mock_security.mbox
```

## 참고

샘플 `.mbox` 파일이 포함되어 있어 로컬에서 바로 기능을 확인할 수 있습니다.
