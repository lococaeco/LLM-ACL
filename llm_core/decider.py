import json
import random
from vllm import SamplingParams
from .loader import LLMLoader

class LLMDecider:
    def __init__(self, model_name: str, use_4bit: bool = False, use_8bit: bool = False, mock: bool = False):
        self.mock = mock
        self.model_name = model_name

        if not self.mock:
            self.loader = LLMLoader(
                model_id=model_name,
                tensor_parallel_size=8,
                gpu_memory_utilization=0.7,
                quantization="fp8" if use_8bit else None
            )
            self.tokenizer, self.model = self.loader.load()
        else:
            self.tokenizer = None
            self.model = None

    def decide_schedule(self, episode_summary: str, action_dim: int, max_steps: int) -> list:
        """
        에피소드 요약을 보고 다음 에피소드의 공격 스케줄을 JSON으로 생성합니다.
        """
        if self.mock:
            return self._mock_schedule(max_steps, action_dim)

        prompt = f"""
System: You are an adversarial attacker testing the robustness of an RL agent.
Your goal is to design a sequence of attacks (Masking, Noise, Scaling) to challenge the agent based on its previous performance.

User:
Here is the summary of the Agent's previous episode:
{episode_summary}

Task:
Create an attack schedule for the next episode (0 to {max_steps} steps).
You can target specific action dimensions (0 to {action_dim-1}).

Attack Types:
- "mask": Set action to 0. (Simulates motor failure)
- "noise": Add random noise. value = std dev (e.g., 0.5).
- "scale": Multiply action. value = factor (e.g., 2.0 for double force, 0.5 for half).
- "none": No attack.

Output Format (JSON only):
[
    {{"start": 0, "end": 100, "type": "none", "target": []}},
    {{"start": 100, "end": 300, "type": "noise", "target": [0, 2], "value": 0.5}},
    {{"start": 300, "end": {max_steps}, "type": "mask", "target": [1], "value": 0}}
]

Response (JSON Only):
"""
        sampling_params = SamplingParams(temperature=0.6, max_tokens=500)
        
        try:
            outputs = self.model.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text
            schedule = self._parse_json(response)
            return schedule
        except Exception as e:
            print(f"[Decider Error] {e}")
            return []

    def _parse_json(self, text: str) -> list:
        """LLM 응답에서 JSON 파싱"""
        try:
            # JSON 부분만 추출 (대괄호 찾기)
            start = text.find('[')
            end = text.rfind(']') + 1
            if start != -1 and end != -1:
                json_str = text[start:end]
                return json.loads(json_str)
        except:
            pass
        return [] # 실패 시 빈 스케줄

    def _mock_schedule(self, max_steps, action_dim):
        """테스트용 랜덤 스케줄"""
        return [
            {"start": 0, "end": max_steps // 2, "type": "none", "target": []},
            {"start": max_steps // 2, "end": max_steps, "type": "noise", "target": [0], "value": 1.0}
        ]