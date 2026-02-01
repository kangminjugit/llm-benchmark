# LLM Benchmark API

LLM 모델의 성능 벤치마크를 측정하는 FastAPI 애플리케이션입니다.

## 주요 기능

- **지연 시간 (Latency)**: 평균, 중앙값, 최소/최대, P95, P99
- **처리량 (Throughput)**: 초당 토큰 수, 초당 요청 수
- **성공률**: 전체 요청 대비 성공한 요청의 비율
- **토큰 사용량**: 총 토큰 수, 요청당 평균 토큰 수
- **비전 모델 지원**: base64 이미지 테스트셋을 랜덤 샘플링하여 벤치마크

## 설치 방법

```bash
pip install -r requirements.txt
```

## 실행 방법

```bash
python llm_benchmark_api.py
```

또는

```bash
uvicorn llm_benchmark_api:app --host 0.0.0.0 --port 8000 --reload
```

## API 사용 예제

`api_endpoint`에는 base URL을 넣으면 `/v1/chat/completions`로 자동 보정됩니다.

### 1. OpenAI API 벤치마크

```bash
curl -X POST "http://localhost:8000/benchmark" \
  -H "Content-Type: application/json" \
  -d '{
    "api_endpoint": "https://api.openai.com",
    "api_key": "your-openai-api-key",
    "model_name": "gpt-3.5-turbo",
    "num_requests": 10
  }'
```

### 2. Anthropic Claude API 벤치마크

```bash
curl -X POST "http://localhost:8000/benchmark" \
  -H "Content-Type: application/json" \
  -d '{
    "api_endpoint": "https://api.anthropic.com",
    "api_key": "your-anthropic-api-key",
    "model_name": "claude-3-sonnet-20240229",
    "num_requests": 10
  }'
```

### 3. 커스텀 프롬프트 사용

```bash
curl -X POST "http://localhost:8000/benchmark" \
  -H "Content-Type: application/json" \
  -d '{
    "api_endpoint": "https://api.openai.com",
    "api_key": "your-api-key",
    "model_name": "gpt-4",
    "test_prompts": [
      "Python으로 퀵소트 알고리즘을 구현해주세요.",
      "데이터베이스 정규화에 대해 설명해주세요.",
      "RESTful API의 설계 원칙은 무엇인가요?"
    ],
    "num_requests": 20
  }'
```

### 4. Python에서 사용

```python
import requests

response = requests.post(
    "http://localhost:8000/benchmark",
    json={
        "api_endpoint": "https://api.openai.com",
        "api_key": "your-api-key",
        "model_name": "gpt-3.5-turbo",
        "num_requests": 10
    }
)

result = response.json()
print(f"평균 지연 시간: {result['metrics']['average_latency']:.2f}초")
print(f"초당 토큰 수: {result['metrics']['average_tokens_per_second']:.2f}")
print(f"성공률: {result['metrics']['success_rate']:.2f}%")
```

### 5. 비전 모델 벤치마크

`data/vision_testset.jsonl`에 base64 이미지가 저장되어 있으며, 요청 시 랜덤으로 샘플링합니다.

```bash
curl -X POST "http://localhost:8000/benchmark" \
  -H "Content-Type: application/json" \
  -d '{
    "api_endpoint": "https://api.openai.com",
    "api_key": "your-api-key",
    "model_name": "gpt-4o-mini",
    "num_requests": 5,
    "use_vision": true,
    "vision_testset_path": "data/vision_testset.jsonl"
  }'
```

## 응답 예제

```json
{
  "model_name": "gpt-3.5-turbo",
  "timestamp": "2026-02-01T10:30:00.000000",
  "metrics": {
    "total_requests": 10,
    "successful_requests": 10,
    "failed_requests": 0,
    "success_rate": 100.0,
    "average_latency": 1.23,
    "median_latency": 1.15,
    "min_latency": 0.89,
    "max_latency": 1.67,
    "p95_latency": 1.58,
    "p99_latency": 1.65,
    "average_tokens_per_second": 45.2,
    "median_tokens_per_second": 44.8,
    "max_tokens_per_second": 52.1,
    "total_tokens_used": 3450,
    "average_tokens_per_request": 345.0,
    "total_duration": 12.3,
    "requests_per_second": 0.81
  },
  "individual_results": [...]
}
```

## 측정 지표 설명

- **average_latency**: 평균 응답 시간 (초)
- **median_latency**: 중앙값 응답 시간 (초)
- **p95_latency**: 95 백분위수 응답 시간 (초)
- **p99_latency**: 99 백분위수 응답 시간 (초)
- **average_tokens_per_second**: 초당 평균 생성 토큰 수
- **requests_per_second**: 초당 처리 가능한 요청 수
- **success_rate**: 성공한 요청의 비율 (%)

## 엔드포인트

- `GET /`: API 정보
- `GET /health`: 헬스 체크
- `POST /benchmark`: 벤치마크 실행

## 주의사항

1. API 키는 안전하게 관리하세요
2. 과도한 요청으로 API 사용량 제한에 걸릴 수 있습니다
3. 큰 `num_requests` 값은 비용이 많이 발생할 수 있습니다
