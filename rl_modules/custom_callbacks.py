"""
ALRT (Adaptive LLM-Guided Robustness Training) 콜백
- 확장된 메트릭 수집 (reward trend, learning phase, robustness score 등)
- 주기적 강건성 평가
- LLM에게 풍부한 상태 정보 제공
"""

import time
from collections import deque
from typing import Any, Dict, List, Optional, Callable

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

from llm_core.decider import LLMDecider


class ALRTCallback(BaseCallback):
    """
    ALRT 메인 콜백: 에이전트 상태 분석 + LLM 의사결정 + 래퍼 플랜 적용
    
    확장된 메트릭:
    - reward_trend: increasing/stable/decreasing/plateau
    - learning_phase: early/mid/late
    - robustness_score: 교란 환경에서의 상대적 성능
    - state_coverage: 방문한 상태 다양성
    """
    
    def __init__(
        self,
        wrapper,
        llm_decider: LLMDecider,
        curriculum_cfg,
        env_spec: Dict[str, Any],
        base_env_fn: Callable = None,  # 강건성 평가용 환경 생성 함수
        target_reward: float = 3000.0,
        total_timesteps: int = 1000000,
        use_llm: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.wrapper = wrapper
        self.llm_decider = llm_decider
        self.cfg = curriculum_cfg
        self.env_spec = env_spec
        self.use_llm = use_llm
        self.base_env_fn = base_env_fn
        self.target_reward = target_reward
        self.total_timesteps_target = total_timesteps
        
        # 에피소드 히스토리
        self.window = deque(maxlen=int(curriculum_cfg.window_size))
        
        # 현재 에피소드 상태
        self.current_reward = 0.0
        self.current_length = 0
        self.current_stage = int(curriculum_cfg.initial_stage)
        self.current_mode = "maintain"  # boost/maintain/perturb
        
        # ALRT 설정
        self.alrt_config = dict(curriculum_cfg.get("alrt", {}))
        self.fail_threshold = float(curriculum_cfg.reward_failure_threshold)
        self.safety_clip = dict(curriculum_cfg.safety_clip)
        
        # 강건성 평가 관련
        self.robustness_eval_freq = self.alrt_config.get("robustness_eval_freq", 20)
        self.robustness_eval_episodes = self.alrt_config.get("robustness_eval_episodes", 3)
        self.last_robustness_score = 1.0
        self.episode_count = 0
        
        # 트렌드 분석용
        self.reward_history = deque(maxlen=50)

    def _on_step(self) -> bool:
        """매 스텝마다 호출"""
        # 보상/길이 누적
        if "rewards" in self.locals:
            self.current_reward += float(self.locals["rewards"][0])
            self.current_length += 1
        
        # 에피소드 종료 처리
        if "dones" in self.locals and self.locals["dones"][0]:
            self._on_episode_end()
        
        return True

    def _on_episode_end(self) -> None:
        """에피소드 종료 시 처리"""
        self.episode_count += 1
        
        # 에피소드 기록
        fail = self.current_reward < self.fail_threshold
        self.window.append({
            "reward": self.current_reward,
            "length": self.current_length,
            "fail": fail,
            "timestep": self.num_timesteps,
        })
        self.reward_history.append(self.current_reward)
        
        # wandb 로깅
        if wandb.run is not None:
            wandb.log({
                "episode_reward": self.current_reward,
                "episode_length": self.current_length,
                "curriculum_stage": self.current_stage,
                "alrt_mode": self.current_mode,
                "timesteps": self.num_timesteps,
            })
        
        # 주기적 강건성 평가
        if self.episode_count % self.robustness_eval_freq == 0:
            self._evaluate_robustness()
        
        # 주기적 LLM 의사결정
        if self.cfg.enabled and self.episode_count % int(self.cfg.llm_update_freq) == 0:
            summary = self._build_extended_summary()
            plan = self._request_plan(summary)
            self.wrapper.apply_plan(plan)
            self.current_stage = plan.get("stage", self.current_stage)
            self.current_mode = plan.get("mode", "maintain")
            
            if wandb.run is not None:
                wandb.log({
                    "alrt_mode": self.current_mode,
                    "curriculum_stage": self.current_stage,
                    "robustness_score": self.last_robustness_score,
                    "reward_trend": summary.get("reward_trend", "unknown"),
                    "learning_phase": summary.get("learning_phase", "unknown"),
                    "plan_source": plan.get("source", "llm"),
                    "timesteps": self.num_timesteps,
                })
        
        # 상태 초기화
        self.current_reward = 0.0
        self.current_length = 0

    def _build_extended_summary(self) -> Dict[str, Any]:
        """LLM에 전달할 확장된 상태 요약 생성"""
        
        rewards = [e["reward"] for e in self.window]
        lengths = [e["length"] for e in self.window]
        fails = [e["fail"] for e in self.window]
        
        # 기본 통계
        mean_reward = float(np.mean(rewards)) if rewards else 0.0
        std_reward = float(np.std(rewards)) if rewards else 0.0
        median_reward = float(np.median(rewards)) if rewards else 0.0
        fail_rate = float(np.mean(fails)) if fails else 0.0
        
        # 트렌드 분석
        reward_trend = self._analyze_reward_trend()
        
        # 학습 단계 판단
        learning_phase = self._determine_learning_phase()
        
        # 상태 커버리지 (래퍼에서 가져옴)
        state_coverage = getattr(self.wrapper, "get_state_coverage", lambda: 0.5)()
        
        # 정책 엔트로피 (가능한 경우)
        policy_entropy = self._get_policy_entropy()
        
        # 목표 대비 진행률
        reward_ratio = mean_reward / self.target_reward if self.target_reward > 0 else 0.0
        
        return {
            # 기본 정보
            "env_name": self.env_spec.get("env_name", "unknown"),
            "window_size": len(self.window),
            "total_episodes": self.episode_count,
            "current_stage": self.current_stage,
            "current_mode": self.current_mode,
            
            # 보상 통계
            "mean_reward": mean_reward,
            "median_reward": median_reward,
            "std_reward": std_reward,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "fail_rate": fail_rate,
            
            # 목표 대비 진행률
            "target_reward": self.target_reward,
            "reward_ratio": reward_ratio,
            
            # 학습 동역학
            "reward_trend": reward_trend,
            "learning_phase": learning_phase,
            "policy_entropy": policy_entropy,
            
            # 강건성 정보
            "robustness_score": self.last_robustness_score,
            
            # 탐험 정보
            "state_coverage": state_coverage,
            "mean_episode_length": float(np.mean(lengths)) if lengths else 0.0,
            
            # 최근 보상 (LLM이 패턴 파악용)
            "recent_rewards": list(rewards)[-10:] if rewards else [],
        }

    def _analyze_reward_trend(self) -> str:
        """보상 트렌드 분석: increasing/stable/decreasing/plateau"""
        
        if len(self.reward_history) < 10:
            return "unknown"
        
        recent = list(self.reward_history)
        n = len(recent)
        
        # 선형 회귀로 기울기 계산
        x = np.arange(n)
        slope = np.polyfit(x, recent, 1)[0]
        
        # 최근 vs 이전 비교
        mid = n // 2
        recent_mean = np.mean(recent[mid:])
        earlier_mean = np.mean(recent[:mid])
        
        # 변화율
        if earlier_mean != 0:
            change_rate = (recent_mean - earlier_mean) / abs(earlier_mean)
        else:
            change_rate = 0
        
        # 분산으로 안정성 체크
        recent_std = np.std(recent[-10:])
        mean_val = np.mean(recent[-10:])
        cv = recent_std / abs(mean_val) if mean_val != 0 else 0  # 변동계수
        
        # 판단
        if abs(change_rate) < 0.05 and cv < 0.15:
            return "plateau"
        elif slope > 0 and change_rate > 0.1:
            return "increasing"
        elif slope < 0 and change_rate < -0.1:
            return "decreasing"
        else:
            return "stable"

    def _determine_learning_phase(self) -> str:
        """학습 단계 판단: early/mid/late"""
        progress = self.num_timesteps / self.total_timesteps_target
        
        if progress < 0.2:
            return "early"
        elif progress < 0.7:
            return "mid"
        else:
            return "late"

    def _get_policy_entropy(self) -> float:
        """정책 엔트로피 추출 (가능한 경우)"""
        try:
            if hasattr(self.model, "ent_coef"):
                return float(self.model.ent_coef)
        except Exception:
            pass
        return -1.0

    def _evaluate_robustness(self) -> None:
        """강건성 평가: 교란 환경에서 현재 정책 성능 측정"""
        
        if self.base_env_fn is None:
            self.last_robustness_score = 0.5
            return
        
        try:
            eval_env = self.base_env_fn()
            
            perturbation_scenarios = [
                {"type": "obs_noise", "value": 0.15},
                {"type": "action_noise", "value": 0.1},
            ]
            
            normal_rewards = []
            perturbed_rewards = []
            
            # 정상 환경에서 평가
            for _ in range(self.robustness_eval_episodes):
                obs, _ = eval_env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = eval_env.step(action)
                    episode_reward += reward
                    done = terminated or truncated
                normal_rewards.append(episode_reward)
            
            # 교란 환경에서 평가
            for scenario in perturbation_scenarios:
                for _ in range(self.robustness_eval_episodes):
                    obs, _ = eval_env.reset()
                    episode_reward = 0
                    done = False
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=True)
                        
                        if scenario["type"] == "obs_noise":
                            obs = obs + np.random.normal(0, scenario["value"], obs.shape)
                        elif scenario["type"] == "action_noise":
                            action = action + np.random.normal(0, scenario["value"], action.shape)
                        
                        obs, reward, terminated, truncated, _ = eval_env.step(action)
                        episode_reward += reward
                        done = terminated or truncated
                    
                    perturbed_rewards.append(episode_reward)
            
            eval_env.close()
            
            normal_mean = np.mean(normal_rewards) if normal_rewards else 1.0
            perturbed_mean = np.mean(perturbed_rewards) if perturbed_rewards else 0.0
            
            if normal_mean > 0:
                self.last_robustness_score = min(1.0, max(0.0, perturbed_mean / normal_mean))
            else:
                self.last_robustness_score = 0.5
            
            if self.verbose:
                print(f"\n[ALRT] 강건성 평가: normal={normal_mean:.1f}, perturbed={perturbed_mean:.1f}, "
                      f"score={self.last_robustness_score:.2f}")
                
        except Exception as e:
            print(f"[ALRT] 강건성 평가 실패: {e}")
            self.last_robustness_score = 0.5

    def _request_plan(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """LLM에게 플랜 요청"""
        
        if not self.use_llm:
            plan = self.llm_decider._mock_plan(
                summary,
                self.cfg.max_plan_horizon,
                self.cfg.max_stage,
                self.alrt_config,
            )
            plan["source"] = "mock"
            return plan
        
        return self.llm_decider.decide_curriculum(
            summary=summary,
            env_spec=self.env_spec,
            stage=self.current_stage,
            safety_clip=self.safety_clip,
            max_horizon=int(self.cfg.max_plan_horizon),
            max_stage=int(self.cfg.max_stage),
            alrt_config=self.alrt_config,
        )


# Legacy alias
LLMCurriculumCallback = ALRTCallback


class WandbLoggingCallback(BaseCallback):
    """wandb에 기본 메트릭을 기록하는 콜백"""
    
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        if "rewards" in self.locals:
            self.current_episode_reward += self.locals["rewards"][0]
            self.current_episode_length += 1

        if "dones" in self.locals and self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            wandb.log({
                "episode_reward": self.current_episode_reward,
                "episode_length": self.current_episode_length,
                "timesteps": self.num_timesteps,
            })
            self.current_episode_reward = 0
            self.current_episode_length = 0

        if self.num_timesteps % self.log_freq == 0:
            self._log_metrics()
        return True

    def _log_metrics(self):
        metrics = {"timesteps": self.num_timesteps}
        if self.episode_rewards:
            metrics["avg_reward_100ep"] = np.mean(self.episode_rewards)
            metrics["avg_len_100ep"] = np.mean(self.episode_lengths)
        wandb.log(metrics)


class ProgressCallback(BaseCallback):
    """학습 진행률/ETA를 출력하는 콜백"""
    
    def __init__(self, total_timesteps: int, update_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.update_freq == 0:
            self._print_progress()
        return True

    def _print_progress(self):
        elapsed = time.time() - self.start_time
        progress = self.num_timesteps / self.total_timesteps if self.total_timesteps else 0
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "█" * filled + "░" * (bar_len - filled)

        if self.num_timesteps > 0:
            eta = elapsed / self.num_timesteps * max(self.total_timesteps - self.num_timesteps, 0)
        else:
            eta = 0

        print(
            f"\rProgress: [{bar}] {progress:.1%} ({self.num_timesteps}/{self.total_timesteps}) "
            f"| Time: {self._fmt(elapsed)} | ETA: {self._fmt(eta)}",
            end="",
            flush=True,
        )
        if self.num_timesteps >= self.total_timesteps:
            print()

    def _fmt(self, seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


class StandardADRCallback(BaseCallback):
    """
    Standard ADR 콜백.
    주기적으로 성능을 체크하고, 목표 성능을 달성하면 랜덤화 범위(phi)를 확장합니다.
    """
    def __init__(
        self,
        env,
        update_freq: int = 2048,
        threshold: float = 0.8,
        target_reward: float = 1000.0,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.env_wrapper = env  # StandardADRWrapper 인스턴스
        self.update_freq = update_freq  # 업데이트 주기 (스텝 수)
        self.threshold_ratio = threshold  # 목표 보상 대비 비율 (예: 0.8)
        self.target_reward = target_reward  # 목표 보상 절대값 (train.py에서 전달받음)
        
        self.reward_buffer = deque(maxlen=50)  # 최근 보상 저장용 버퍼
        self.last_update_step = 0
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        # 보상 수집
        if "rewards" in self.locals:
             self.current_reward += float(self.locals["rewards"][0])

        # 에피소드 종료 시 버퍼에 추가
        if "dones" in self.locals and self.locals["dones"][0]:
            self.reward_buffer.append(self.current_reward)
            self.current_reward = 0.0
            
        # 주기적으로 업데이트 체크
        if self.num_timesteps - self.last_update_step >= self.update_freq:
            self.last_update_step = self.num_timesteps
            
            if len(self.reward_buffer) > 0:
                mean_reward = np.mean(self.reward_buffer)
                target = self.target_reward * self.threshold_ratio
                
                # 평균 보상이 목표치를 넘으면 범위 확장
                if mean_reward >= target:
                    self.env_wrapper.expand_bounds()
                    if self.verbose > 0:
                        print(f"[ADR] 성능 달성 ({mean_reward:.2f} >= {target:.2f}). 범위를 확장합니다.")
                
                # WandB 로깅
                if wandb.run is not None:
                    wandb.log({
                        "adr/phi": self.env_wrapper.get_phi(),
                        "adr/mean_reward": mean_reward,
                        "adr/target_threshold": target
                    }, step=self.num_timesteps)
                    
        return True