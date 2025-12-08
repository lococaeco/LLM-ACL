"""
ALRT (Adaptive LLM-Guided Robustness Training) 의사결정 클래스
- 에이전트 상태 분석
- BOOST/MAINTAIN/PERTURB 모드 결정
- 교란/강화 플랜 생성
"""

import json
import os
import re
import textwrap
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from vllm import SamplingParams
from .loader import LLMLoader


class ALRTMode(Enum):
    BOOST = "boost"
    MAINTAIN = "maintain"
    PERTURB = "perturb"


class LLMDecider:
    """
    ALRT 의사결정 클래스
    에피소드 통계를 분석하여 BOOST/MAINTAIN/PERTURB 모드 결정
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
        self.current_mode = ALRTMode.MAINTAIN
        self.decision_history: List[Dict] = []
        
        # 로깅 설정
        self.log_raw = bool(self.prompt_cfg.get("log_raw", False))
        default_dir = os.path.join(os.path.dirname(__file__), "..", "outputs", "llm_logs")
        self.log_dir = self.prompt_cfg.get("log_dir") or default_dir
        if self.log_raw:
            os.makedirs(self.log_dir, exist_ok=True)
        
        # LLM 로드
        if not self.mock:
            quantization = "fp8" if use_8bit else None
            self.loader = LLMLoader(
                model_id=model_name,
                tensor_parallel_size=8,
                gpu_memory_utilization=0.85,
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
        alrt_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """ALRT 메인 의사결정 함수"""
        
        if self.mock:
            return self._mock_plan(summary, max_horizon, max_stage, alrt_config)
        
        prompt = self._build_alrt_prompt(summary, env_spec, stage, max_horizon, alrt_config)
        
        # Stop 토큰 최소화 - JSON이 완성될 때까지 생성하도록
        sampling_params = SamplingParams(
            temperature=float(self.prompt_cfg.get("temperature", 0.4)),
            top_p=float(self.prompt_cfg.get("top_p", 0.9)),
            max_tokens=int(self.prompt_cfg.get("max_length", 512)),
            # stop 토큰 제거하거나 최소화
        )
        
        try:
            outputs = self.model.generate([prompt], sampling_params)
            text = outputs[0].outputs[0].text
            
            if self.log_raw:
                self._log_raw_output(prompt, text)
            
            plan = self._parse_and_validate(text, env_spec, safety_clip, max_horizon, max_stage, alrt_config)
            
            # 모드 전환 기록
            new_mode = ALRTMode(plan.get("mode", "maintain"))
            if new_mode != self.current_mode:
                print(f"\n[ALRT] 모드 전환: {self.current_mode.value} → {new_mode.value}")
                print(f"       이유: {plan.get('reasoning', 'N/A')}")
            self.current_mode = new_mode
            
            return plan
            
        except Exception as e:
            print(f"[Decider Error] {e}")
            return self._mock_plan(summary, max_horizon, max_stage, alrt_config)

    def _build_alrt_prompt(
        self,
        summary: Dict[str, Any],
        env_spec: Dict[str, Any],
        stage: int,
        max_horizon: int,
        alrt_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """ALRT용 프롬프트 생성 - 풍부한 컨텍스트 제공"""
        
        alrt_config = alrt_config or {}
        env_name = env_spec.get("env_name", "unknown")
        target_reward = alrt_config.get("target_rewards", {}).get(env_name, 3000)
        
        # 핵심 메트릭 추출
        mean_reward = summary.get("mean_reward", 0)
        std_reward = summary.get("std_reward", 0)
        min_reward = summary.get("min_reward", 0)
        max_reward = summary.get("max_reward", 0)
        fail_rate = summary.get("fail_rate", 0)
        learning_phase = summary.get("learning_phase", "early")
        reward_trend = summary.get("reward_trend", "unknown")
        robustness_score = summary.get("robustness_score", 1.0)
        total_episodes = summary.get("total_episodes", 0)
        recent_rewards = summary.get("recent_rewards", [])
        
        # 최근 에피소드 상세 정보 (LLM이 패턴을 볼 수 있도록)
        recent_str = ""
        if recent_rewards:
            recent_str = "RECENT EPISODE REWARDS (oldest to newest):\n"
            for i, r in enumerate(recent_rewards):
                if i > 0:
                    if r > recent_rewards[i-1]:
                        marker = "[UP]"
                    elif r < recent_rewards[i-1]:
                        marker = "[DOWN]"
                    else:
                        marker = "[SAME]"
                else:
                    marker = ""
                recent_str += f"  Episode {total_episodes - len(recent_rewards) + i + 1}: {r:>8.1f} {marker}\n"
        
        # 진행 상황 계산
        reward_ratio = mean_reward / target_reward if target_reward > 0 else 0
        progress_bar = self._make_progress_bar(reward_ratio)
        
        # 모드 결정 힌트 (LLM 가이드용)
        if mean_reward < target_reward * 0.3 or fail_rate > 0.5 or learning_phase == "early":
            suggested = "boost"
            reason_hint = "Agent is struggling or in early phase"
        elif (mean_reward > target_reward * 0.7 and robustness_score < 0.6) or reward_trend == "plateau":
            suggested = "perturb"
            reason_hint = "Agent performing well but needs robustness training"
        elif reward_trend == "increasing":
            suggested = "maintain"
            reason_hint = "Agent is improving, do not interfere"
        else:
            suggested = "maintain"
            reason_hint = "Default to observation mode"
        
        prompt = f'''You are an RL training coach for {env_name}. Analyze and decide the training mode.

=== CURRENT STATUS ===
Total Episodes: {total_episodes}
Learning Phase: {learning_phase}
Target Reward: {target_reward}

Performance:
  Mean Reward: {mean_reward:>8.1f} ({reward_ratio*100:>5.1f}% of target)
  Std Reward:  {std_reward:>8.1f}
  Min/Max:     {min_reward:>8.1f} / {max_reward:>8.1f}
  Progress:    {progress_bar}

Health Indicators:
  Fail Rate:        {fail_rate*100:>5.1f}%
  Reward Trend:     {reward_trend}
  Robustness Score: {robustness_score:.2f} (1.0 = untested, <0.6 = needs training)

{recent_str}
=== DECISION RULES ===
BOOST:    mean_reward < {target_reward * 0.3:.0f} OR fail_rate > 50% OR early phase. Help agent learn.
MAINTAIN: reward_trend == "increasing". Do not interfere with learning.
PERTURB:  mean_reward > {target_reward * 0.7:.0f} with low robustness, OR plateau. Add perturbations.

=== YOUR TASK ===
Suggested mode: {suggested} ({reason_hint})

Output a JSON object with your decision. You may adjust the suggestion if you disagree.
For BOOST: set reward_scale (1.0-1.5) and exploration_bonus (0.0-0.2)
For PERTURB: set episode_plan with types: obs_noise, action_noise, action_delay

Example JSON format:
{{"mode": "boost", "reasoning": "agent struggling in early phase", "stage": 0, "boost_config": {{"reward_scale": 1.2, "exploration_bonus": 0.1}}, "perturb_config": {{"episode_plan": []}}}}

Your JSON response:'''

        return prompt

    def _make_progress_bar(self, ratio: float, length: int = 20) -> str:
        """진행률 바 생성"""
        ratio = max(0, min(1, ratio))
        filled = int(length * ratio)
        bar = "█" * filled + "░" * (length - filled)
        return f"[{bar}] {ratio*100:.1f}%"

    def _parse_and_validate(
        self,
        text: str,
        env_spec: Dict[str, Any],
        safety: Dict[str, float],
        max_horizon: int,
        max_stage: int,
        alrt_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """LLM 응답에서 첫 번째 유효한 JSON 추출 및 검증"""
        
        # 1. JSON 추출 시도
        json_obj = self._extract_first_json(text)
        
        if json_obj is None:
            print(f"[ALRT] JSON 파싱 실패, fallback 사용")
            return self._fallback_plan(max_horizon, max_stage)
        
        # 2. 모드 검증
        mode = json_obj.get("mode", "maintain").lower().strip()
        if mode not in ["boost", "maintain", "perturb"]:
            mode = "maintain"
        
        result = {
            "mode": mode,
            "source": "llm",
            "reasoning": json_obj.get("reasoning", ""),
            "stage": min(max(int(json_obj.get("stage", 0)), 0), max_stage),
        }
        
        # 3. 모드별 설정 추출 및 검증
        if mode == "boost":
            boost_cfg = json_obj.get("boost_config", {})
            result["boost_config"] = {
                "reward_scale": min(max(float(boost_cfg.get("reward_scale", 1.2)), 1.0), 1.5),
                "exploration_bonus": min(max(float(boost_cfg.get("exploration_bonus", 0.1)), 0.0), 0.2),
            }
            result["perturb_config"] = {"episode_plan": []}
            
        elif mode == "perturb":
            perturb_cfg = json_obj.get("perturb_config", {})
            episode_plan = self._validate_episode_plan(
                perturb_cfg.get("episode_plan", []),
                safety, max_horizon
            )
            result["perturb_config"] = {"episode_plan": episode_plan}
            result["boost_config"] = {"reward_scale": 1.0, "exploration_bonus": 0.0}
            
        else:  # maintain
            result["boost_config"] = {"reward_scale": 1.0, "exploration_bonus": 0.0}
            result["perturb_config"] = {"episode_plan": []}
        
        return result

    def _extract_first_json(self, text: str) -> Optional[Dict]:
        """텍스트에서 첫 번째 유효한 JSON 객체 추출 (빈 값 무시)"""
        
        if not text:
            return None
        
        # 중괄호 매칭으로 JSON 후보 찾기
        candidates = []
        stack = []
        start_idx = None
        
        for i, ch in enumerate(text):
            if ch == "{":
                if not stack:
                    start_idx = i
                stack.append("{")
            elif ch == "}" and stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidates.append(text[start_idx:i+1])
        
        # 각 후보를 파싱하고, 유효한 값이 있는 것만 반환
        for candidate in candidates:
            try:
                fixed = self._fix_incomplete_json(candidate)
                obj = json.loads(fixed)
                
                if not isinstance(obj, dict):
                    continue
                
                # "mode" 키가 있고, 값이 유효한지 확인
                mode = obj.get("mode", "")
                if not mode or not mode.strip():
                    continue  # 빈 mode는 스킵
                
                # mode가 유효한 값인지 확인
                if mode.lower().strip() in ["boost", "maintain", "perturb"]:
                    return obj
                    
            except:
                continue
        
        # 유효한 JSON을 못 찾은 경우
        return None

    def _fix_incomplete_json(self, text: str) -> str:
        """불완전한 JSON 수정 시도"""
        text = text.strip()
        
        # 끝나지 않은 문자열 닫기
        if text.count('"') % 2 == 1:
            text += '"'
        
        # 닫히지 않은 괄호 닫기
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        if open_brackets > 0:
            text += ']' * open_brackets
        if open_braces > 0:
            text += '}' * open_braces
        
        return text

    def _validate_episode_plan(
        self,
        plan: List[Dict],
        safety: Dict[str, float],
        max_horizon: int,
    ) -> List[Dict]:
        """episode_plan 검증 및 안전 범위 클리핑"""
        
        validated = []
        for seg in plan:
            try:
                attack_type = seg.get("type", "none")
                if attack_type == "none":
                    continue
                
                start = max(0, int(seg.get("start", 0)))
                end = min(max_horizon, int(seg.get("end", max_horizon)))
                if end <= start:
                    continue
                
                value = float(seg.get("value", 0.0))
                target = seg.get("target", [])
                
                # 안전 범위 클리핑
                if attack_type == "obs_noise":
                    value = min(abs(value), safety.get("obs_noise_std", 0.3))
                elif attack_type == "action_noise":
                    value = min(abs(value), safety.get("action_noise_std", 0.3))
                elif attack_type == "action_delay":
                    value = min(max(0, int(value)), int(safety.get("action_delay_max", 3)))
                elif attack_type == "obs_dropout":
                    value = min(max(0, value), safety.get("obs_dropout", 0.2))
                
                validated.append({
                    "start": start,
                    "end": end,
                    "type": attack_type,
                    "target": target if isinstance(target, list) else [],
                    "value": value,
                })
            except:
                continue
        
        return validated

    def _fallback_plan(self, max_horizon: int, max_stage: int) -> Dict[str, Any]:
        """파싱 실패 시 기본 플랜"""
        return {
            "mode": "maintain",
            "source": "fallback",
            "reasoning": "JSON parsing failed, defaulting to maintain",
            "stage": 0,
            "boost_config": {"reward_scale": 1.0, "exploration_bonus": 0.0},
            "perturb_config": {"episode_plan": []},
        }

    def _mock_plan(
        self,
        summary: Dict[str, Any],
        max_horizon: int,
        max_stage: int,
        alrt_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Mock 모드: 규칙 기반 의사결정"""
        
        alrt_config = alrt_config or {}
        mean_reward = summary.get("mean_reward", 0)
        fail_rate = summary.get("fail_rate", 0)
        learning_phase = summary.get("learning_phase", "early")
        reward_trend = summary.get("reward_trend", "unknown")
        robustness_score = summary.get("robustness_score", 1.0)
        target_reward = summary.get("target_reward", 3000)
        
        # 규칙 기반 모드 결정
        if mean_reward < target_reward * 0.3 or fail_rate > 0.5 or learning_phase == "early":
            mode = "boost"
            reasoning = f"Agent struggling (reward={mean_reward:.0f}, fail_rate={fail_rate:.1%})"
            boost_config = {"reward_scale": 1.2, "exploration_bonus": 0.1}
            perturb_config = {"episode_plan": []}
            
        elif (mean_reward > target_reward * 0.7 and robustness_score < 0.6) or reward_trend == "plateau":
            mode = "perturb"
            reasoning = f"Agent ready for robustness training (robustness={robustness_score:.2f})"
            boost_config = {"reward_scale": 1.0, "exploration_bonus": 0.0}
            # 점진적 교란 플랜
            mid = max_horizon // 2
            perturb_config = {
                "episode_plan": [
                    {"start": 0, "end": mid, "type": "obs_noise", "target": [], "value": 0.1},
                    {"start": mid, "end": max_horizon, "type": "action_noise", "target": [], "value": 0.08},
                ]
            }
        else:
            mode = "maintain"
            reasoning = f"Agent learning well (trend={reward_trend})"
            boost_config = {"reward_scale": 1.0, "exploration_bonus": 0.0}
            perturb_config = {"episode_plan": []}
        
        return {
            "mode": mode,
            "source": "mock",
            "reasoning": reasoning,
            "stage": 0,
            "boost_config": boost_config,
            "perturb_config": perturb_config,
        }

    def _log_raw_output(self, prompt: str, raw: str) -> None:
        """LLM 입출력 로깅"""
        parsed = self._extract_first_json(raw)
        pretty = json.dumps(parsed, ensure_ascii=False, indent=2) if parsed else raw
        
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = os.path.join(self.log_dir, f"llm_output_{stamp}.txt")
        
        with open(fname, "w", encoding="utf-8") as f:
            f.write("=== PROMPT ===\n")
            f.write(prompt)
            f.write("\n\n=== RAW OUTPUT ===\n")
            f.write(raw)
            f.write("\n\n=== PARSED ===\n")
            f.write(pretty)
        
        print(f"\n[LLM] Mode decision logged to {fname}")