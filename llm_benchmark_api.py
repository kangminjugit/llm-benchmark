from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import time
import asyncio
from datetime import datetime
import statistics

app = FastAPI(
    title="LLM Benchmark API",
    description="API endpoint로 LLM 모델의 성능 벤치마크를 측정합니다",
    version="1.0.0"
)

class ModelRequest(BaseModel):
    api_endpoint: str
    api_key: str
    model_name: str
    test_prompts: Optional[List[str]] = None
    num_requests: Optional[int] = 10
    
class BenchmarkResponse(BaseModel):
    model_name: str
    timestamp: str
    metrics: Dict[str, Any]
    individual_results: List[Dict[str, Any]]

class LLMBenchmark:
    """LLM 모델 벤치마크 측정 클래스"""
    
    def __init__(self, api_endpoint: str, api_key: str, model_name: str):
        self.api_endpoint = self._normalize_chat_completions_endpoint(api_endpoint)
        self.api_key = api_key
        self.model_name = model_name
        self.default_prompts = [
            "안녕하세요. 인공지능에 대해 간단히 설명해주세요.",
            "Python에서 리스트와 튜플의 차이점을 설명해주세요.",
            "기후 변화가 환경에 미치는 영향은 무엇인가요?",
            "머신러닝과 딥러닝의 차이점을 설명해주세요.",
            "블록체인 기술의 원리를 간단히 설명해주세요."
        ]

    @staticmethod
    def _normalize_chat_completions_endpoint(api_endpoint: str) -> str:
        """Chat Completions API 엔드포인트로 정규화"""
        endpoint = api_endpoint.rstrip("/")
        if endpoint.endswith("/v1/chat/completions") or endpoint.endswith("/chat/completions"):
            return endpoint
        if endpoint.endswith("/v1"):
            return f"{endpoint}/chat/completions"
        return f"{endpoint}/v1/chat/completions"
    
    async def make_api_call(self, prompt: str) -> Dict[str, Any]:
        """API 호출 및 응답 시간 측정"""
        import httpx
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # OpenAI 호환 API 형식 사용
                response = await client.post(
                    self.api_endpoint,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 500
                    }
                )
                
                end_time = time.time()
                latency = end_time - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # 응답 추출
                    if "choices" in data and len(data["choices"]) > 0:
                        completion_text = data["choices"][0]["message"]["content"]
                        
                        # 토큰 정보 추출
                        usage = data.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        total_tokens = usage.get("total_tokens", 0)
                        
                        return {
                            "success": True,
                            "latency": latency,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens,
                            "response_length": len(completion_text),
                            "tokens_per_second": completion_tokens / latency if latency > 0 else 0,
                            "error": None
                        }
                    else:
                        return {
                            "success": False,
                            "latency": latency,
                            "error": "Invalid response format"
                        }
                else:
                    return {
                        "success": False,
                        "latency": latency,
                        "error": f"HTTP {response.status_code}: {response.text}"
                    }
                    
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "latency": end_time - start_time,
                "error": str(e)
            }
    
    async def run_benchmark(self, prompts: List[str], num_requests: int) -> Dict[str, Any]:
        """벤치마크 실행"""
        
        # 사용할 프롬프트 결정
        test_prompts = prompts if prompts else self.default_prompts
        
        # 반복 실행을 위한 프롬프트 리스트 생성
        all_prompts = []
        for i in range(num_requests):
            all_prompts.append(test_prompts[i % len(test_prompts)])
        
        # 모든 요청 동시 실행
        print(f"총 {len(all_prompts)}개의 요청 실행 중...")
        results = await asyncio.gather(*[
            self.make_api_call(prompt) for prompt in all_prompts
        ])
        
        # 성공한 결과만 필터링
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]
        
        if not successful_results:
            raise Exception("모든 요청이 실패했습니다")
        
        # 지표 계산
        latencies = [r["latency"] for r in successful_results]
        tokens_per_second = [r["tokens_per_second"] for r in successful_results]
        total_tokens_list = [r["total_tokens"] for r in successful_results]
        
        metrics = {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "success_rate": len(successful_results) / len(results) * 100,
            
            # 지연 시간 (Latency)
            "average_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "p95_latency": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 1 else latencies[0],
            "p99_latency": statistics.quantiles(latencies, n=100)[98] if len(latencies) > 1 else latencies[0],
            
            # 처리량 (Throughput)
            "average_tokens_per_second": statistics.mean(tokens_per_second),
            "median_tokens_per_second": statistics.median(tokens_per_second),
            "max_tokens_per_second": max(tokens_per_second),
            
            # 토큰 사용량
            "total_tokens_used": sum(total_tokens_list),
            "average_tokens_per_request": statistics.mean(total_tokens_list),
            
            # 전체 처리량
            "total_duration": sum(latencies),
            "requests_per_second": len(successful_results) / sum(latencies) if sum(latencies) > 0 else 0,
        }
        
        return {
            "metrics": metrics,
            "individual_results": results
        }

@app.get("/")
async def root():
    """API 루트"""
    return {
        "message": "LLM Benchmark API",
        "endpoints": {
            "/benchmark": "POST - 벤치마크 실행",
            "/health": "GET - 헬스 체크"
        }
    }

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/benchmark", response_model=BenchmarkResponse)
async def run_benchmark(request: ModelRequest):
    """
    LLM 모델 벤치마크 실행
    
    Parameters:
    - api_endpoint: LLM API base URL 또는 Chat Completions 엔드포인트 (예: https://api.openai.com 또는 https://api.openai.com/v1/chat/completions)
    - api_key: API 키
    - model_name: 모델 이름 (예: gpt-3.5-turbo)
    - test_prompts: 테스트용 프롬프트 리스트 (선택사항)
    - num_requests: 실행할 요청 수 (기본값: 10)
    
    Returns:
    - 벤치마크 지표 및 개별 결과
    """
    try:
        benchmark = LLMBenchmark(
            api_endpoint=request.api_endpoint,
            api_key=request.api_key,
            model_name=request.model_name
        )
        
        result = await benchmark.run_benchmark(
            prompts=request.test_prompts,
            num_requests=request.num_requests
        )
        
        return BenchmarkResponse(
            model_name=request.model_name,
            timestamp=datetime.now().isoformat(),
            metrics=result["metrics"],
            individual_results=result["individual_results"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
