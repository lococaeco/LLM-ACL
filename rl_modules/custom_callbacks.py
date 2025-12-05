"""
커스텀 콜백 모듈
RL 학습 중 LLM을 호출하여 Robustness 제어를 수행하는 콜백들을 제공합니다.
"""

import numpy as np
import wandb
import time
from stable_baselines3.common.callbacks import BaseCallback
from llm_core import LLMDecider


class LLMControlCallback(BaseCallback):
    """
    학습 중 주기적으로 LLM을 호출하여 Action Dropout Mask를 업데이트하는 콜백

    이 콜백은 학습 진행 상황(보상, 손실 등)을 텍스트로 요약하여 LLM에 전달하고,
    반환된 마스크를 환경 래퍼에 적용하여 에이전트의 Robustness를 동적으로 제어합니다.
    """

    def __init__(self, wrapper, llm_decider: LLMDecider, update_freq: int = 1000, verbose: int = 0):
        """
        LLMControlCallback 초기화

        Args:
            wrapper: 업데이트할 LLMGuidedRobustnessWrapper 객체
            llm_decider (LLMDecider): LLM 의사결정 객체
            update_freq (int): LLM 호출 주기 (스텝 단위)
            verbose (int): 상세 출력 레벨
        """
        super().__init__(verbose)
        self.wrapper = wrapper
        self.llm_decider = llm_decider
        self.update_freq = update_freq
        self.last_update = 0
        self.rewards = []  # 최근 보상 기록
        self.losses = []   # 최근 손실 기록

    def _on_step(self) -> bool:
        """
        각 스텝마다 호출되는 메서드
        보상과 손실을 기록하고, 주기마다 LLM을 호출하여 마스크를 업데이트합니다.

        Returns:
            bool: 학습 계속 여부
        """
        # 보상 기록 (가능한 경우)
        if 'rewards' in self.locals:
            self.rewards.extend(self.locals['rewards'])

        # 손실 기록 (가능한 경우)
        if 'loss' in self.locals:
            self.losses.append(self.locals['loss'])

        # 업데이트 주기 확인 및 LLM 호출
        if self.num_timesteps - self.last_update >= self.update_freq:
            self._update_mask()
            self.last_update = self.num_timesteps

        return True

    def _update_mask(self):
        """
        현재 학습 상태를 요약하여 LLM에 전달하고, 반환된 마스크로 래퍼를 업데이트합니다.
        """
        # 학습 상태 텍스트 요약
        avg_reward = np.mean(self.rewards[-1000:]) if self.rewards else 0.0
        avg_loss = np.mean(self.losses[-100:]) if self.losses else 0.0

        state_text = f"""
                            RL 에이전트 학습 상태:
                            - 평균 보상 (최근 1000개): {avg_reward:.2f}
                            - 평균 손실 (최근 100개): {avg_loss:.4f}
                            - 총 학습 스텝: {self.num_timesteps}
                            - 현재 에피소드: {self.n_calls if hasattr(self, 'n_calls') else '알 수 없음'}
                    """

        # LLM을 통한 마스크 결정
        action_dim = self.wrapper.action_dim
        mask = self.llm_decider.decide_action_dropout(state_text, action_dim)

        # 래퍼에 마스크 적용
        self.wrapper.update_mask(mask)

        # 상세 출력
        if self.verbose > 0:
            print(f"[LLM 콜백] 마스크 업데이트: {mask}")
            print(f"[LLM 콜백] 상태 요약: 평균 보상={avg_reward:.2f}, 평균 손실={avg_loss:.4f}")


class WandbLoggingCallback(BaseCallback):
    """
    wandb에 학습 메트릭을 기록하는 콜백
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        """
        WandbLoggingCallback 초기화

        Args:
            log_freq (int): 로그 기록 주기 (스텝 단위)
            verbose (int): 상세 출력 레벨
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        """
        각 스텝마다 호출되는 메서드
        지정된 주기마다 메트릭을 wandb에 기록합니다.

        Returns:
            bool: 학습 계속 여부
        """
        # 스텝별 보상 누적
        if 'rewards' in self.locals and len(self.locals['rewards']) > 0:
            self.current_episode_reward += self.locals['rewards'][0]
            self.current_episode_length += 1

        # 에피소드 종료 시 기록
        if 'dones' in self.locals and len(self.locals['dones']) > 0 and self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # wandb에 에피소드 보상 기록
            wandb.log({
                'episode_reward': self.current_episode_reward,
                'episode_length': self.current_episode_length,
                'timesteps': self.num_timesteps
            })
            
            # 초기화
            self.current_episode_reward = 0
            self.current_episode_length = 0

        # 로그 주기마다 추가 메트릭 기록
        if self.num_timesteps % self.log_freq == 0:
            self._log_metrics()

        return True

    def _log_metrics(self):
        """
        현재 메트릭을 wandb에 기록합니다.
        """
        metrics = {
            'timesteps': self.num_timesteps,
        }

        # 평균 메트릭 계산 및 기록
        if self.episode_rewards:
            metrics['avg_reward'] = np.mean(self.episode_rewards[-10:])  # 최근 10개 에피소드 평균
        if self.episode_lengths:
            metrics['avg_episode_length'] = np.mean(self.episode_lengths[-10:])

        # 학습 관련 메트릭 (가능한 경우)
        if 'loss' in self.locals:
            metrics['loss'] = self.locals['loss']

        # wandb에 기록
        wandb.log(metrics)

        if self.verbose > 0:
            print(f"[Wandb] 메트릭 기록: {metrics}")


class ProgressCallback(BaseCallback):
    """
    학습 진행 상황을 시각적으로 표시하는 콜백
    """

    def __init__(self, total_timesteps: int, update_freq: int = 1000, verbose: int = 1):
        """
        ProgressCallback 초기화

        Args:
            total_timesteps (int): 총 학습 스텝 수
            update_freq (int): 진행 상황 업데이트 주기
            verbose (int): 상세 출력 레벨
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_timesteps = 0

    def _on_step(self) -> bool:
        """
        각 스텝마다 호출되는 메서드
        진행 상황을 계산하여 출력합니다.

        Returns:
            bool: 학습 계속 여부
        """
        if self.num_timesteps % self.update_freq == 0:
            self._print_progress()
        
        return True

    def _print_progress(self):
        """
        현재 진행 상황을 계산하여 출력합니다.
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        progress = self.num_timesteps / self.total_timesteps
        
        # 예상 남은 시간 계산
        if self.num_timesteps > 0:
            avg_time_per_step = elapsed_time / self.num_timesteps
            remaining_steps = self.total_timesteps - self.num_timesteps
            eta = avg_time_per_step * remaining_steps
        else:
            eta = 0
        
        # 진행 바 생성
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # 시간 포맷팅
        elapsed_str = self._format_time(elapsed_time)
        eta_str = self._format_time(eta)
        
        # 출력
        progress_str = (
            f"\r진행률: [{bar}] {progress:.1%} "
            f"({self.num_timesteps}/{self.total_timesteps}) "
            f"경과시간: {elapsed_str} | 예상남은시간: {eta_str}"
        )
        
        print(progress_str, end='', flush=True)
        
        # 100% 완료 시 줄바꿈
        if self.num_timesteps >= self.total_timesteps:
            print()

    def _format_time(self, seconds: float) -> str:
        """
        시간을 hh:mm:ss 형식으로 포맷팅합니다.
        """
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return "02d"