import os
import random

from locust import HttpUser, between, task


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class BenchmarkUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self) -> None:
        self.api_endpoint = os.getenv("BENCHMARK_API_ENDPOINT", "https://api.openai.com")
        self.api_key = os.getenv("BENCHMARK_API_KEY", "")
        self.model_name = os.getenv("BENCHMARK_MODEL_NAME", "gpt-3.5-turbo")
        self.num_requests = int(os.getenv("BENCHMARK_NUM_REQUESTS", "1"))
        self.use_vision = _parse_bool(os.getenv("BENCHMARK_USE_VISION", "false"))
        self.vision_testset_path = os.getenv(
            "BENCHMARK_VISION_TESTSET_PATH", "data/vision_testset.jsonl"
        )
        self.test_prompts = [
            prompt.strip()
            for prompt in os.getenv("BENCHMARK_TEST_PROMPTS", "").split("||")
            if prompt.strip()
        ]

    @task
    def run_benchmark(self) -> None:
        payload = {
            "api_endpoint": self.api_endpoint,
            "api_key": self.api_key,
            "model_name": self.model_name,
            "num_requests": self.num_requests,
            "use_vision": self.use_vision,
        }
        if self.test_prompts:
            payload["test_prompts"] = [random.choice(self.test_prompts)]
        if self.use_vision:
            payload["vision_testset_path"] = self.vision_testset_path

        self.client.post("/benchmark", json=payload, name="/benchmark")
