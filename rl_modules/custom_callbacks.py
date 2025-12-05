import numpy as np
import wandb
import time
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from llm_core.decider import LLMDecider

class LLMControlCallback(BaseCallback):
    """
    에피소드 데이터를 수집하고, 에피소드 종료 시 LLM에게 공격 스케줄을 요청하는 콜백
    """
    # [수정 1] __init__에 max_episode_steps 인자 추가 (기본값 1000)
    def __init__(self, wrapper, llm_decider: LLMDecider, max_episode_steps: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.wrapper = wrapper
        self.llm_decider = llm_decider
        self.max_episode_steps = max_episode_steps  # 변수에 저장
        
        # 에피소드 데이터 버퍼
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        
    def _on_step(self) -> bool:
        """매 스텝 데이터 수집"""
        obs = self.locals.get('new_obs')
        if obs is not None:
            self.obs_buffer.append(np.mean(obs)) 
        
        actions = self.locals.get('actions')
        if actions is not None:
            self.action_buffer.append(actions[0])
            
        rewards = self.locals.get('rewards')
        if rewards is not None:
            self.reward_buffer.append(rewards[0])
            
        dones = self.locals.get('dones')
        if dones and dones[0]:
            self._on_episode_end()
            
        return True

    def _on_episode_end(self):
        """에피소드가 끝나면 요약 후 LLM 호출"""
        summary_text = self._summarize_episode()
        
        # [수정 2] 환경에서 조회하지 않고 저장된 값 사용 (에러 원인 제거)
        new_schedule = self.llm_decider.decide_schedule(
            episode_summary=summary_text,
            action_dim=self.wrapper.action_dim,
            max_steps=self.max_episode_steps
        )
        
        if new_schedule:
            self.wrapper.update_schedule(new_schedule)
            
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []

    def _summarize_episode(self) -> str:
        """1 에피소드 데이터를 4분할하여 통계 요약"""
        length = len(self.reward_buffer)
        if length == 0: return "No data."
        
        chunk_size = max(1, length // 4)
        summary_parts = []
        
        for i in range(0, length, chunk_size):
            end = min(i + chunk_size, length)
            r_chunk = self.reward_buffer[i:end]
            a_chunk = np.array(self.action_buffer[i:end])
            
            avg_rew = np.mean(r_chunk)
            action_var = np.var(a_chunk, axis=0).mean()
            
            summary_parts.append(
                f"- Step {i}~{end}: Avg Reward={avg_rew:.2f}, Action Variance={action_var:.2f}"
            )
            
        return "\n".join(summary_parts)


class WandbLoggingCallback(BaseCallback):
    """
    wandb에 학습 메트릭을 기록하는 콜백
    """
    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        # 보상 누적
        if 'rewards' in self.locals:
            reward = self.locals['rewards'][0]
            self.current_episode_reward += reward
            self.current_episode_length += 1

        # 에피소드 종료 확인
        if 'dones' in self.locals and self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            wandb.log({
                'episode_reward': self.current_episode_reward,
                'episode_length': self.current_episode_length,
                'timesteps': self.num_timesteps
            })
            
            self.current_episode_reward = 0
            self.current_episode_length = 0

        # 주기적 로그 기록
        if self.num_timesteps % self.log_freq == 0:
            self._log_metrics()

        return True

    def _log_metrics(self):
        metrics = {'timesteps': self.num_timesteps}
        
        if self.episode_rewards:
            metrics['avg_reward_100ep'] = np.mean(self.episode_rewards)
            metrics['avg_len_100ep'] = np.mean(self.episode_lengths)
            
        if 'loss' in self.locals:
            metrics['loss'] = self.locals['loss']

        wandb.log(metrics)


class ProgressCallback(BaseCallback):
    """
    학습 진행 상황을 시각적으로 표시하는 콜백
    """
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
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        progress = self.num_timesteps / self.total_timesteps if self.total_timesteps > 0 else 0
        
        # ETA 계산
        if self.num_timesteps > 0:
            avg_time_per_step = elapsed_time / self.num_timesteps
            remaining_steps = self.total_timesteps - self.num_timesteps
            eta = avg_time_per_step * remaining_steps
        else:
            eta = 0
        
        # 진행 바
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        elapsed_str = self._format_time(elapsed_time)
        eta_str = self._format_time(eta)
        
        # \r을 사용하여 같은 줄에 덮어쓰기
        print(f"\rProcess: [{bar}] {progress:.1%} ({self.num_timesteps}/{self.total_timesteps}) "
              f"| Time: {elapsed_str} | ETA: {eta_str}", end='', flush=True)
        
        if self.num_timesteps >= self.total_timesteps:
            print() # 완료 시 줄바꿈

    def _format_time(self, seconds: float) -> str:
        """초 단위 시간을 hh:mm:ss 문자열로 변환 (버그 수정됨)"""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"