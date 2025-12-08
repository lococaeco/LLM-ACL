import time
from collections import deque
from typing import Any, Dict, List

import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback

from llm_core.decider import LLMDecider


class LLMCurriculumCallback(BaseCallback):
    """
    에피소드 통계를 모아 주기적으로 LLM에 커리큘럼 플랜을 요청하고,
    환경 래퍼에 적용하는 콜백.
    """
    def __init__(
        self,
        wrapper,
        llm_decider: LLMDecider,
        curriculum_cfg,
        env_spec: Dict[str, Any],
        use_llm: bool = True,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.wrapper = wrapper
        self.llm_decider = llm_decider
        self.cfg = curriculum_cfg
        self.env_spec = env_spec
        self.use_llm = use_llm
        self.window = deque(maxlen=int(curriculum_cfg.window_size))
        self.current_reward = 0.0
        self.current_length = 0
        self.current_stage = int(curriculum_cfg.initial_stage)
        self.fail_threshold = float(curriculum_cfg.reward_failure_threshold)
        self.safety_clip = dict(curriculum_cfg.safety_clip)

    def _on_step(self) -> bool:
        # 스텝 보상/길이 누적
        if "rewards" in self.locals:
            self.current_reward += float(self.locals["rewards"][0])
            self.current_length += 1

        # 에피소드 종료 시점 처리
        if "dones" in self.locals and self.locals["dones"][0]:
            self._on_episode_end()

        return True

    def _on_episode_end(self):
        # 에피소드 기록 저장
        fail = self.current_reward < self.fail_threshold
        self.window.append(
            {"reward": self.current_reward, "length": self.current_length, "fail": fail}
        )

        # wandb 로깅
        if wandb.run is not None:
            wandb.log(
                {
                    "episode_reward": self.current_reward,
                    "episode_length": self.current_length,
                    "curriculum_stage": self.current_stage,
                    "timesteps": self.num_timesteps,
                }
            )

        # 주기적으로 LLM 호출
        ep_count = len(self.window)
        if self.cfg.enabled and ep_count % int(self.cfg.llm_update_freq) == 0:
            summary = self._build_summary()
            plan = self._request_plan(summary)
            self.wrapper.apply_plan(plan)
            self.current_stage = plan.get("stage", self.current_stage)

            if wandb.run is not None:
                wandb.log(
                    {
                        "curriculum_stage": self.current_stage,
                        "plan_segments": len(plan.get("episode_plan", [])),
                        "timesteps": self.num_timesteps,
                    }
                )

        # 에피소드 상태 초기화
        self.current_reward = 0.0
        self.current_length = 0

    # -------- 내부 유틸 --------

    def _build_summary(self) -> Dict[str, Any]:
        rewards = [e["reward"] for e in self.window]
        lengths = [e["length"] for e in self.window]
        fails = [e["fail"] for e in self.window]

        return {
            "window_size": len(self.window),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "median_reward": float(np.median(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "fail_rate": float(np.mean(fails)) if fails else 0.0,
            "recent_rewards": rewards,
            "current_stage": self.current_stage,
        }

    def _request_plan(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        if not self.use_llm:
            # LLM 비활성 시 mock 플랜 사용
            return self.llm_decider._mock_plan(self.cfg.max_plan_horizon, self.cfg.max_stage)

        return self.llm_decider.decide_curriculum(
            summary=summary,
            env_spec=self.env_spec,
            stage=self.current_stage,
            safety_clip=self.safety_clip,
            max_horizon=int(self.cfg.max_plan_horizon),
            max_stage=int(self.cfg.max_stage),
        )


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
            wandb.log(
                {
                    "episode_reward": self.current_episode_reward,
                    "episode_length": self.current_episode_length,
                    "timesteps": self.num_timesteps,
                }
            )
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
        if "loss" in self.locals:
            metrics["loss"] = self.locals["loss"]
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
            f"\rProcess: [{bar}] {progress:.1%} ({self.num_timesteps}/{self.total_timesteps}) "
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