import numpy as np
import gymnasium as gym
from gymnasium import ActionWrapper
from collections import deque


class LLMGuidedRobustnessWrapper(ActionWrapper):
    """
    (이전 버전) 액션 마스킹/노이즈만 수행하던 래퍼.
    호환성을 위해 남겨두지만 새 커리큘럼 래퍼 사용을 권장.
    """
    def __init__(self, env, action_dim: int):
        super().__init__(env)
        self.action_dim = action_dim
        self.current_step = 0
        self.attack_schedule = []

    def update_schedule(self, schedule: list):
        self.attack_schedule = schedule
        print(f"[Wrapper] 새로운 공격 스케줄 적용됨 (총 {len(schedule)}개 구간)")

    def reset(self, **kwargs):
        self.current_step = 0
        return super().reset(**kwargs)

    def action(self, action):
        modified_action = action.copy()
        for plan in self.attack_schedule:
            if plan['start'] <= self.current_step < plan['end']:
                modified_action = self._apply_attack(modified_action, plan)
        self.current_step += 1
        return modified_action

    def _apply_attack(self, action, plan):
        attack_type = plan.get('type', 'none')
        targets = plan.get('target', [])
        val = plan.get('value', 0.0)
        for idx in targets:
            if idx >= self.action_dim:
                continue
            if attack_type == 'mask':
                action[idx] = 0.0
            elif attack_type == 'noise':
                noise = np.random.normal(0, val)
                action[idx] += noise
            elif attack_type == 'scale':
                action[idx] *= val
            elif attack_type == 'fix':
                action[idx] = val
        return action


class CurriculumEnvWrapper(gym.Wrapper):
    """
    LLM 커리큘럼 플랜을 받아 관측/행동/동역학 교란을 적용하는 래퍼
    - obs_noise, obs_dropout, action_noise, action_scale, action_delay, dyn_shift 지원
    """
    def __init__(self, env: gym.Env, max_episode_steps: int, safety_clip: dict):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.safety_clip = safety_clip
        self.current_step = 0
        self.plan = {"episode_plan": [], "stage": 0}
        self._delay_buffer = deque()
        self._max_delay = int(safety_clip.get("action_delay_max", 0))
        self._last_delay_action = None  # 지연 시 사용할 기본값 저장

    def apply_plan(self, plan: dict):
        """LLM 출력 플랜을 환경에 적용"""
        self.plan = plan or {"episode_plan": [], "stage": 0}
        self._delay_buffer.clear()
        self._last_delay_action = None
        print(f"[Curriculum] 새 플랜 적용: 세그먼트 {len(self.plan.get('episode_plan', []))}개, stage={self.plan.get('stage', 0)}")

    def reset(self, **kwargs):
        self.current_step = 0
        self._delay_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def step(self, action):
        # 행동 변환 적용
        effective_action = self._apply_action_transforms(np.array(action, copy=True))
        obs, reward, terminated, truncated, info = self.env.step(effective_action)
        # 관측 변환 적용
        obs = self._apply_obs_transforms(np.array(obs, copy=True))
        self.current_step += 1
        return obs, reward, terminated, truncated, info

    # ---------- 내부 동작 ----------

    def _active_segments(self):
        return [
            seg for seg in self.plan.get("episode_plan", [])
            if seg.get("start", 0) <= self.current_step < seg.get("end", self.max_episode_steps)
        ]

    def _apply_action_transforms(self, action: np.ndarray) -> np.ndarray:
        if action is None:
            return action
        active = self._active_segments()

        # 지연 계산을 먼저 수행해 기준 액션을 만든다
        delay = 0
        for seg in active:
            if seg.get("type") == "action_delay":
                delay = max(delay, int(seg.get("value", 0)))
        if delay > 0:
            action = self._delay_action(action, delay)

        for seg in active:
            atype = seg.get("type")
            target = seg.get("target", [])
            val = seg.get("value", 0.0)

            if atype == "action_noise":
                self._apply_noise(action, target, val)
            elif atype == "action_scale":
                self._apply_scale(action, target, val)
            elif atype == "dyn_shift":
                # 실제 동역학 변경은 환경마다 달라 범용적으로 적용하기 어렵다.
                # 여기서는 존재할 경우 속성에 값을 반영하고, 없으면 무시.
                params = seg.get("params", {})
                self._apply_dyn_shift(params)

        return action

    def _apply_obs_transforms(self, obs: np.ndarray) -> np.ndarray:
        active = self._active_segments()
        for seg in active:
            otype = seg.get("type")
            target = seg.get("target", [])
            val = seg.get("value", 0.0)

            if otype == "obs_noise":
                self._apply_noise(obs, target, val)
            elif otype == "obs_dropout":
                self._apply_dropout(obs, target, val)

        return obs

    def _apply_noise(self, arr: np.ndarray, target: list, std: float):
        """지정 인덱스(없으면 전체)에 가우시안 노이즈 추가"""
        idx = target if target else range(arr.shape[-1])
        for i in idx:
            if 0 <= i < arr.shape[-1]:
                arr[i] += np.random.normal(0, std)

    def _apply_scale(self, arr: np.ndarray, target: list, scale: float):
        """지정 인덱스(없으면 전체)에 스케일 적용"""
        idx = target if target else range(arr.shape[-1])
        for i in idx:
            if 0 <= i < arr.shape[-1]:
                arr[i] *= scale

    def _apply_dropout(self, arr: np.ndarray, target: list, rate: float):
        """확률적으로 관측을 0으로 드롭"""
        idx = target if target else range(arr.shape[-1])
        for i in idx:
            if 0 <= i < arr.shape[-1] and np.random.rand() < rate:
                arr[i] = 0.0

    def _delay_action(self, action: np.ndarray, delay: int) -> np.ndarray:
        """
        액션 지연: 최근 액션을 큐에 쌓고, delay 만큼 전의 액션을 사용.
        초기에는 0 벡터를 사용.
        """
        if self._last_delay_action is None:
            self._last_delay_action = np.zeros_like(action)
        self._delay_buffer.append(np.array(action, copy=True))
        if len(self._delay_buffer) > delay:
            delayed = self._delay_buffer[-(delay + 1)]
            self._last_delay_action = delayed
            return delayed
        return self._last_delay_action

    def _apply_dyn_shift(self, params: dict):
        """
        동역학 파라미터 변경 시도 (환경별 지원 여부 다름)
        - MuJoCo 계열: env.model 속성에 friction/mass_scale가 있다면 반영
        - 지원하지 않으면 조용히 무시
        """
        friction = params.get("friction")
        mass_scale = params.get("mass_scale")

        model = getattr(self.env, "model", None)
        if model is not None:
            try:
                if friction is not None and hasattr(model, "geom_friction"):
                    model.geom_friction[:] *= friction
                if mass_scale is not None and hasattr(model, "body_mass"):
                    model.body_mass[:] *= mass_scale
            except Exception:
                # 환경이 지원하지 않으면 무시
                pass