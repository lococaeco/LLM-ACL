"""
ALRT (Adaptive LLM-Guided Robustness Training) 환경 래퍼
- BOOST 모드: reward shaping, exploration bonus 적용
- MAINTAIN 모드: 기본 환경 유지
- PERTURB 모드: 관측/행동/동역학 교란 적용
"""

import numpy as np
import gymnasium as gym
from gymnasium import ActionWrapper
from collections import deque
from typing import Any, Dict, Optional, Tuple


class LLMGuidedRobustnessWrapper(ActionWrapper):
    """
    (Legacy) 이전 버전 호환용. 새 코드는 ALRTEnvWrapper 사용 권장.
    """
    def __init__(self, env, action_dim: int):
        super().__init__(env)
        self.action_dim = action_dim
        self.current_step = 0
        self.attack_schedule = []

    def update_schedule(self, schedule: list):
        self.attack_schedule = schedule

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
                action[idx] += np.random.normal(0, val)
            elif attack_type == 'scale':
                action[idx] *= val
        return action


class ALRTEnvWrapper(gym.Wrapper):
    """
    ALRT 환경 래퍼: LLM 판단에 따라 BOOST/MAINTAIN/PERTURB 적용
    
    Features:
    - BOOST 모드: reward scaling, exploration bonus 적용
    - MAINTAIN 모드: 환경 그대로 유지
    - PERTURB 모드: obs_noise, action_noise, action_delay 등 교란
    """
    
    def __init__(
        self,
        env: gym.Env,
        max_episode_steps: int,
        safety_clip: Dict[str, float],
    ):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.safety_clip = safety_clip
        
        # 상태 변수
        self.current_step = 0
        self.episode_count = 0
        
        # ALRT 플랜
        self.plan: Dict[str, Any] = {
            "mode": "maintain",
            "boost_config": {"reward_scale": 1.0, "exploration_bonus": 0.0},
            "perturb_config": {"episode_plan": []},
            "stage": 0,
        }
        
        # Action delay 버퍼
        self._delay_buffer = deque()
        self._last_action = None
        self._max_delay = int(safety_clip.get("action_delay_max", 3))
        
        # 에피소드 내 통계 (exploration bonus 계산용)
        self._visited_states = set()
        self._state_discretization = 10  # 상태 이산화 해상도

    def apply_plan(self, plan: Dict[str, Any]) -> None:
        """LLM 출력 플랜 적용"""
        self.plan = plan or {
            "mode": "maintain",
            "boost_config": {"reward_scale": 1.0, "exploration_bonus": 0.0},
            "perturb_config": {"episode_plan": []},
            "stage": 0,
        }
        self._delay_buffer.clear()
        self._last_action = None
        
        mode = self.plan.get("mode", "maintain")
        source = self.plan.get("source", "llm")
        reasoning = self.plan.get("reasoning", "")
        
        print(f"\n[ALRT] 플랜 적용: mode={mode}, stage={self.plan.get('stage', 0)}, source={source}")
        if reasoning:
            print(f"       이유: {reasoning}")

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        """환경 리셋"""
        self.current_step = 0
        self.episode_count += 1
        self._delay_buffer.clear()
        self._last_action = None
        self._visited_states.clear()
        
        obs, info = self.env.reset(**kwargs)
        
        # 상태 방문 기록
        self._record_state_visit(obs)
        
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """환경 스텝 - 모드에 따라 다른 처리"""
        
        mode = self.plan.get("mode", "maintain")
        
        # 1. 행동 변환 (PERTURB 모드)
        if mode == "perturb":
            action = self._apply_action_perturbations(np.array(action, copy=True))
        
        # 2. 환경 스텝 실행
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 3. 관측 변환 (PERTURB 모드)
        if mode == "perturb":
            obs = self._apply_obs_perturbations(np.array(obs, copy=True))
        
        # 4. 보상 변환 (BOOST 모드)
        if mode == "boost":
            reward = self._apply_boost_reward(reward, obs, info)
        
        # 상태 방문 기록
        self._record_state_visit(obs)
        
        self.current_step += 1
        
        # 디버그 정보 추가
        info["alrt_mode"] = mode
        info["alrt_stage"] = self.plan.get("stage", 0)
        
        return obs, reward, terminated, truncated, info

    # ========== BOOST 모드 함수들 ==========
    
    def _apply_boost_reward(self, reward: float, obs: np.ndarray, info: Dict) -> float:
        """BOOST 모드: 보상 스케일링 + 탐험 보너스"""
        
        boost_cfg = self.plan.get("boost_config", {})
        reward_scale = boost_cfg.get("reward_scale", 1.0)
        exploration_bonus_coef = boost_cfg.get("exploration_bonus", 0.0)
        
        # 1. 기본 보상 스케일링
        scaled_reward = reward * reward_scale
        
        # 2. 탐험 보너스 (새로운 상태 방문 시)
        exploration_bonus = 0.0
        if exploration_bonus_coef > 0:
            state_key = self._discretize_state(obs)
            if state_key not in self._visited_states:
                # 새로운 상태 방문 보너스
                exploration_bonus = exploration_bonus_coef
        
        return scaled_reward + exploration_bonus

    def _discretize_state(self, obs: np.ndarray) -> tuple:
        """상태를 이산화하여 해시 가능한 키로 변환"""
        discretized = np.round(obs * self._state_discretization) / self._state_discretization
        return tuple(discretized.flatten())

    def _record_state_visit(self, obs: np.ndarray) -> None:
        """상태 방문 기록"""
        state_key = self._discretize_state(obs)
        self._visited_states.add(state_key)

    # ========== PERTURB 모드 함수들 ==========
    
    def _get_active_perturbations(self) -> list:
        """현재 스텝에서 활성화된 교란 세그먼트 반환"""
        perturb_cfg = self.plan.get("perturb_config", {})
        episode_plan = perturb_cfg.get("episode_plan", [])
        
        return [
            seg for seg in episode_plan
            if seg.get("start", 0) <= self.current_step < seg.get("end", self.max_episode_steps)
        ]

    def _apply_action_perturbations(self, action: np.ndarray) -> np.ndarray:
        """행동에 교란 적용"""
        if action is None:
            return action
        
        active = self._get_active_perturbations()
        
        # 1. Action delay 처리 (먼저)
        max_delay = 0
        for seg in active:
            if seg.get("type") == "action_delay":
                max_delay = max(max_delay, int(seg.get("value", 0)))
        
        if max_delay > 0:
            action = self._apply_action_delay(action, max_delay)
        
        # 2. 다른 교란들 적용
        for seg in active:
            seg_type = seg.get("type", "none")
            target = seg.get("target", [])
            value = seg.get("value", 0.0)
            params = seg.get("params", {})
            
            if seg_type == "action_noise":
                self._add_noise(action, target, value)
            elif seg_type == "action_scale":
                self._apply_scale(action, target, value)
            elif seg_type == "dyn_shift":
                self._apply_dynamics_shift(params)
        
        return action

    def _apply_obs_perturbations(self, obs: np.ndarray) -> np.ndarray:
        """관측에 교란 적용"""
        active = self._get_active_perturbations()
        
        for seg in active:
            seg_type = seg.get("type", "none")
            target = seg.get("target", [])
            value = seg.get("value", 0.0)
            
            if seg_type == "obs_noise":
                self._add_noise(obs, target, value)
            elif seg_type == "obs_dropout":
                self._apply_dropout(obs, target, value)
        
        return obs

    def _add_noise(self, arr: np.ndarray, target: list, std: float) -> None:
        """가우시안 노이즈 추가 (in-place)"""
        indices = target if target else range(arr.shape[-1])
        for i in indices:
            if 0 <= i < arr.shape[-1]:
                arr[i] += np.random.normal(0, std)

    def _apply_scale(self, arr: np.ndarray, target: list, scale: float) -> None:
        """스케일 적용 (in-place)"""
        indices = target if target else range(arr.shape[-1])
        for i in indices:
            if 0 <= i < arr.shape[-1]:
                arr[i] *= scale

    def _apply_dropout(self, arr: np.ndarray, target: list, rate: float) -> None:
        """확률적 드롭아웃 (in-place)"""
        indices = target if target else range(arr.shape[-1])
        for i in indices:
            if 0 <= i < arr.shape[-1] and np.random.rand() < rate:
                arr[i] = 0.0

    def _apply_action_delay(self, action: np.ndarray, delay: int) -> np.ndarray:
        """행동 지연 적용"""
        if self._last_action is None:
            self._last_action = np.zeros_like(action)
        
        self._delay_buffer.append(np.array(action, copy=True))
        
        if len(self._delay_buffer) > delay:
            delayed = self._delay_buffer[0]
            self._delay_buffer.popleft()
            self._last_action = delayed
            return delayed
        
        return self._last_action.copy()

    def _apply_dynamics_shift(self, params: Dict) -> None:
        """동역학 파라미터 변경 (MuJoCo 환경용)"""
        friction = params.get("friction")
        mass_scale = params.get("mass_scale")
        
        model = getattr(self.env.unwrapped, "model", None)
        if model is None:
            return
        
        try:
            if friction is not None and hasattr(model, "geom_friction"):
                # 상대적 변화가 아닌 절대값으로 설정하도록 수정 필요
                pass  # MuJoCo 환경별 구현 필요
            if mass_scale is not None and hasattr(model, "body_mass"):
                pass  # MuJoCo 환경별 구현 필요
        except Exception:
            pass

    # ========== 유틸리티 ==========
    
    def get_state_coverage(self) -> float:
        """상태 공간 커버리지 추정 (0~1)"""
        # 방문한 이산화 상태 수 / 예상 최대 상태 수
        # 이는 휴리스틱한 추정치
        visited = len(self._visited_states)
        # 간단히 로그 스케일로 정규화
        coverage = min(1.0, np.log1p(visited) / 10.0)
        return coverage


# Legacy alias for backward compatibility
AdaptiveCurriculumWrapper = ALRTEnvWrapper
CurriculumEnvWrapper = ALRTEnvWrapper


class StandardADRWrapper(gym.Wrapper):
    """
    Standard ADR (Automatic Domain Randomization) Wrapper.
    물리 파라미터(마찰, 질량)를 동적 범위(phi) 내에서 랜덤화합니다.
    """
    def __init__(self, env: gym.Env, initial_range: float = 0.0, expand_rate: float = 0.05):
        super().__init__(env)
        self.phi = initial_range  # 현재 랜덤화 범위 (0.0 ~ 1.0+)
        self.expand_rate = expand_rate  # 범위 확장 비율
        
        # 원본 파라미터 저장 (이 값을 기준으로 랜덤화 수행)
        self.original_params = {}
        self._save_original_params()

    def _save_original_params(self):
        """MuJoCo 모델의 원본 물리 파라미터를 저장합니다."""
        try:
            # MuJoCo 모델 객체 접근
            model = self.env.unwrapped.model
            
            if hasattr(model, "geom_friction"):
                self.original_params["friction"] = model.geom_friction.copy()
            
            if hasattr(model, "body_mass"):
                self.original_params["body_mass"] = model.body_mass.copy()
                
        except AttributeError:
            print("[ADR] 경고: MuJoCo 모델 파라미터에 접근할 수 없습니다.")

    def reset(self, **kwargs):
        """환경 리셋 시 파라미터 랜덤화 적용"""
        self._randomize_parameters()
        return self.env.reset(**kwargs)

    def _randomize_parameters(self):
        """물리 파라미터에 랜덤 노이즈를 적용합니다."""
        if self.phi <= 0:
            return

        try:
            model = self.env.unwrapped.model
            
            # 마찰 계수 랜덤화
            if "friction" in self.original_params:
                orig_fric = self.original_params["friction"]
                # 범위: [1-phi, 1+phi] (최소 0.1 보장)
                low = max(0.1, 1.0 - self.phi)
                high = 1.0 + self.phi
                scale = np.random.uniform(low, high, size=orig_fric.shape)
                model.geom_friction[:] = orig_fric * scale

            # 질량 랜덤화
            if "body_mass" in self.original_params:
                orig_mass = self.original_params["body_mass"]
                low = max(0.1, 1.0 - self.phi)
                high = 1.0 + self.phi
                scale = np.random.uniform(low, high, size=orig_mass.shape)
                model.body_mass[:] = orig_mass * scale
                
        except AttributeError:
            pass

    def expand_bounds(self):
        """랜덤화 범위를 확장합니다."""
        self.phi += self.expand_rate
        print(f"[ADR] 범위 확장됨: phi = {self.phi:.4f}")
        
    def get_phi(self):
        """현재 phi 값 반환"""
        return self.phi