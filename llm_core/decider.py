import json
import textwrap
from typing import Any, Dict, List

from vllm import SamplingParams
from .loader import LLMLoader


class LLMDecider:
    """
    에피소드 집계 요약을 받아 적대적 커리큘럼 플랜(JSON)을 생성하는 클래스
    vLLM 로더는 그대로 사용하며, mock 모드에서는 안전한 기본 플랜을 반환.
    """
    def __init__(
        self,
        model_name: str,
        use_4bit: bool = False,
        use_8bit: bool = False,
        mock: bool = False,
        prompt_cfg: Dict[str, Any] | None = None,
    ):
        self.mock = mock
        self.prompt_cfg = prompt_cfg or {}
        self.model_name = model_name

        if not self.mock:
            # vLLM 양자화 설정: 8bit만 fp8로 전달, 4bit는 미지원이므로 기본값 유지
            quantization = "fp8" if use_8bit else None
            self.loader = LLMLoader(
                model_id=model_name,
                tensor_parallel_size=8,
                gpu_memory_utilization=0.7,
                quantization=quantization,
            )
            self.tokenizer, self.model = self.loader.load()
        else:
            self.loader = None
            self.tokenizer = None
            self.model = None

    def decide_curriculum(
        self,
        summary: Dict[str, Any],
        env_spec: Dict[str, Any],
        stage: int,
        safety_clip: Dict[str, float],
        max_horizon: int,
        max_stage: int,
    ) -> Dict[str, Any]:
        """
        요약/스펙/현재 단계 정보를 받아 다음 에피소드 플랜을 생성.
        """
        if self.mock:
            return self._mock_plan(max_horizon, max_stage)

        prompt = self._build_prompt(summary, env_spec, stage, max_horizon)
        sampling_params = SamplingParams(
            temperature=float(self.prompt_cfg.get("temperature", 0.6)),
            top_p=float(self.prompt_cfg.get("top_p", 0.9)),
            max_tokens=int(self.prompt_cfg.get("max_length", 512)),
        )

        try:
            outputs = self.model.generate([prompt], sampling_params)
            text = outputs[0].outputs[0].text
            return self._parse_and_clip(text, env_spec, safety_clip, max_horizon, max_stage)
        except Exception as e:
            print(f"[Decider Error] {e}")
            return self._mock_plan(max_horizon, max_stage)

    # ---------------- 내부 유틸 ----------------

    def _build_prompt(self, summary: Dict[str, Any], env_spec: Dict[str, Any], stage: int, max_horizon: int) -> str:
        """
        프롬프트를 구조화해 길이를 줄이고 재현성을 높인다.
        """
        summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
        env_text = json.dumps(env_spec, ensure_ascii=False, indent=2)

        return textwrap.dedent(
            f"""
            System: 너는 RL 에이전트를 시험하는 적대적 커리큘럼 디자이너다. 현실적 범위 내에서 점진적으로 난이도를 올려라.
            제안은 항상 JSON 스키마만 반환할 것.
            
            [환경 스펙]
            {env_text}

            [현재 단계] {stage}  (0=쉬움, {env_spec.get('max_stage', 3)}=최대 난이도)
            [에피소드 길이 한계] 0~{max_horizon} 스텝

            [최근 에피소드 요약]
            {summary_text}

            [출력 스키마(JSON)]
            {{
              "stage": <int>,
              "episode_plan": [
                {{"start":0,"end":300,"type":"obs_noise","target":[0,1],"value":0.05}},
                {{"start":300,"end":700,"type":"action_delay","value":2}},
                {{"start":700,"end":{max_horizon},"type":"dyn_shift","params":{{"friction":0.9,"mass_scale":1.05}}}}
              ],
              "stage_escalate_if": {{"median_reward_gt": 1800, "fail_rate_lt": 0.1}},
              "stage_deescalate_if": {{"median_reward_lt": 1200}}
            }}

            type은 다음 중 하나: obs_noise, obs_dropout, action_noise, action_scale, action_delay, dyn_shift, none
            target은 관측/액션 인덱스 리스트(없으면 전체 적용). 안전/물리 한계를 넘지 말 것.
            응답은 JSON만 반환하라.
            """
        ).strip()

    def _parse_and_clip(
        self,
        text: str,
        env_spec: Dict[str, Any],
        safety: Dict[str, float],
        max_horizon: int,
        max_stage: int,
    ) -> Dict[str, Any]:
        """LLM 응답을 JSON으로 파싱하고 안전 범위로 클리핑"""
        plan = self._parse_json(text)
        if not isinstance(plan, dict):
            return self._mock_plan(max_horizon, max_stage)

        cleaned = {"episode_plan": []}

        # 단계 정보 클리핑
        stage = plan.get("stage", 0)
        cleaned["stage"] = int(min(max(stage, 0), max_stage))

        act_dim = env_spec.get("act_dim")
        obs_dim = env_spec.get("obs_dim")

        for seg in plan.get("episode_plan", []):
            try:
                start = max(0, int(seg.get("start", 0)))
                end = min(max_horizon, int(seg.get("end", max_horizon)))
                if end < start:
                    end = start
                attack_type = seg.get("type", "none")
                target = [int(t) for t in seg.get("target", []) if isinstance(t, (int, float))]
                target = [t for t in target if t >= 0]
                value = float(seg.get("value", 0.0))
                params = seg.get("params", {})

                # 안전 범위 클리핑
                if attack_type == "obs_noise":
                    value = min(abs(value), safety.get("obs_noise_std", 0.0))
                elif attack_type == "obs_dropout":
                    value = min(max(value, 0.0), safety.get("obs_dropout", 0.0))
                elif attack_type == "action_noise":
                    value = min(abs(value), safety.get("action_noise_std", 0.0))
                elif attack_type == "action_scale":
                    lo = safety.get("action_scale_min", 1.0)
                    hi = safety.get("action_scale_max", 1.0)
                    value = max(lo, min(value, hi))
                elif attack_type == "action_delay":
                    value = min(max(0, int(value)), int(safety.get("action_delay_max", 0)))
                elif attack_type == "dyn_shift":
                    params = {
                        "friction": max(safety.get("friction_min", 0.0), min(params.get("friction", 1.0), safety.get("friction_max", 1.0))),
                        "mass_scale": max(safety.get("mass_scale_min", 1.0), min(params.get("mass_scale", 1.0), safety.get("mass_scale_max", 1.0))),
                    }

                cleaned["episode_plan"].append(
                    {"start": start, "end": end, "type": attack_type, "target": target, "value": value, "params": params}
                )
            except Exception:
                # 잘못된 세그먼트는 무시
                continue

        cleaned["stage_escalate_if"] = plan.get("stage_escalate_if", {})
        cleaned["stage_deescalate_if"] = plan.get("stage_deescalate_if", {})

        return cleaned if cleaned["episode_plan"] else self._mock_plan(max_horizon, max_stage)

    def _parse_json(self, text: str) -> Any:
        """대괄호/중괄호 위치를 찾아 JSON 부분만 파싱"""
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end != -1:
                return json.loads(text[start:end])
        except Exception:
            pass
        return None

    def _mock_plan(self, max_horizon: int, max_stage: int) -> Dict[str, Any]:
        """LLM 없이도 파이프라인을 테스트하기 위한 안전 플랜"""
        mid = max_horizon // 2
        return {
            "stage": 1,
            "episode_plan": [
                {"start": 0, "end": mid, "type": "obs_noise", "target": [], "value": 0.05},
                {"start": mid, "end": max_horizon, "type": "action_noise", "target": [], "value": 0.1},
            ],
            "stage_escalate_if": {"median_reward_gt": 1800, "fail_rate_lt": 0.1},
            "stage_deescalate_if": {"median_reward_lt": 1200},
        }