"""
LLM Benchmark API 테스트 클라이언트
"""
import requests
import json
from typing import Dict, Any

class BenchmarkClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def run_benchmark(
        self,
        api_endpoint: str,
        api_key: str,
        model_name: str,
        test_prompts: list = None,
        num_requests: int = 10
    ) -> Dict[str, Any]:
        """벤치마크 실행"""
        
        payload = {
            "api_endpoint": api_endpoint,
            "api_key": api_key,
            "model_name": model_name,
            "num_requests": num_requests
        }
        
        if test_prompts:
            payload["test_prompts"] = test_prompts
        
        response = requests.post(
            f"{self.base_url}/benchmark",
            json=payload,
            timeout=300  # 5분 타임아웃
        )
        
        response.raise_for_status()
        return response.json()
    
    def print_results(self, result: Dict[str, Any]):
        """결과를 보기 좋게 출력"""
        print("\n" + "="*60)
        print(f"모델: {result['model_name']}")
        print(f"측정 시간: {result['timestamp']}")
        print("="*60)
        
        metrics = result['metrics']
        
        print("\n📊 전체 통계")
        print(f"  총 요청 수: {metrics['total_requests']}")
        print(f"  성공: {metrics['successful_requests']}")
        print(f"  실패: {metrics['failed_requests']}")
        print(f"  성공률: {metrics['success_rate']:.2f}%")
        
        print("\n⏱️  지연 시간 (Latency)")
        print(f"  평균: {metrics['average_latency']:.3f}초")
        print(f"  중앙값: {metrics['median_latency']:.3f}초")
        print(f"  최소: {metrics['min_latency']:.3f}초")
        print(f"  최대: {metrics['max_latency']:.3f}초")
        print(f"  P95: {metrics['p95_latency']:.3f}초")
        print(f"  P99: {metrics['p99_latency']:.3f}초")
        
        print("\n🚀 처리량 (Throughput)")
        print(f"  평균 토큰/초: {metrics['average_tokens_per_second']:.2f}")
        print(f"  중앙값 토큰/초: {metrics['median_tokens_per_second']:.2f}")
        print(f"  최대 토큰/초: {metrics['max_tokens_per_second']:.2f}")
        print(f"  요청/초: {metrics['requests_per_second']:.3f}")
        
        print("\n🎯 토큰 사용량")
        print(f"  총 토큰: {metrics['total_tokens_used']:,}")
        print(f"  평균 토큰/요청: {metrics['average_tokens_per_request']:.1f}")
        print(f"  총 소요 시간: {metrics['total_duration']:.2f}초")
        
        print("\n" + "="*60 + "\n")

def example_openai():
    """OpenAI API 벤치마크 예제"""
    client = BenchmarkClient()
    
    print("OpenAI API 벤치마크를 실행합니다...")
    
    result = client.run_benchmark(
        api_endpoint="https://api.openai.com/v1/chat/completions",
        api_key="your-openai-api-key-here",  # 실제 API 키로 변경
        model_name="gpt-3.5-turbo",
        num_requests=5
    )
    
    client.print_results(result)

def example_custom_prompts():
    """커스텀 프롬프트 벤치마크 예제"""
    client = BenchmarkClient()
    
    custom_prompts = [
        "Python에서 리스트 컴프리헨션을 설명해주세요.",
        "데이터베이스 인덱스의 장단점은 무엇인가요?",
        "Docker와 가상 머신의 차이점을 설명해주세요.",
    ]
    
    print("커스텀 프롬프트로 벤치마크를 실행합니다...")
    
    result = client.run_benchmark(
        api_endpoint="https://api.openai.com/v1/chat/completions",
        api_key="your-api-key-here",  # 실제 API 키로 변경
        model_name="gpt-3.5-turbo",
        test_prompts=custom_prompts,
        num_requests=9  # 각 프롬프트 3번씩
    )
    
    client.print_results(result)

def compare_models():
    """여러 모델 비교 예제"""
    client = BenchmarkClient()
    
    models = [
        ("gpt-3.5-turbo", "https://api.openai.com/v1/chat/completions"),
        ("gpt-4", "https://api.openai.com/v1/chat/completions"),
    ]
    
    results = []
    
    for model_name, endpoint in models:
        print(f"\n{model_name} 벤치마크 실행 중...")
        try:
            result = client.run_benchmark(
                api_endpoint=endpoint,
                api_key="your-api-key-here",  # 실제 API 키로 변경
                model_name=model_name,
                num_requests=5
            )
            results.append((model_name, result))
            client.print_results(result)
        except Exception as e:
            print(f"❌ {model_name} 벤치마크 실패: {e}")
    
    # 비교 결과 출력
    if len(results) > 1:
        print("\n" + "="*60)
        print("모델 비교")
        print("="*60)
        print(f"{'모델':<20} {'평균 지연시간':<15} {'토큰/초':<15} {'성공률':<10}")
        print("-"*60)
        
        for model_name, result in results:
            metrics = result['metrics']
            print(f"{model_name:<20} "
                  f"{metrics['average_latency']:.3f}초{'':<8} "
                  f"{metrics['average_tokens_per_second']:.2f}{'':<8} "
                  f"{metrics['success_rate']:.1f}%")

if __name__ == "__main__":
    print("LLM Benchmark 테스트 클라이언트\n")
    
    # 예제 실행 (주석 해제하여 사용)
    # example_openai()
    # example_custom_prompts()
    # compare_models()
    
    print("사용 방법:")
    print("1. llm_benchmark_api.py를 먼저 실행하세요")
    print("2. 이 파일의 예제 함수에서 API 키를 설정하세요")
    print("3. 원하는 예제 함수의 주석을 해제하고 실행하세요")
