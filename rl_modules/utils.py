"""
RL 환경 생성 및 유틸리티 함수들
"""

import gymnasium as gym
from gymnasium.wrappers import TimeLimit


def make_env(env_name: str, render_mode: str = None, max_episode_steps: int = None) -> gym.Env:
    """
    지정된 환경을 생성하고 필요한 래퍼를 적용하는 함수
    """
    env = gym.make(env_name, render_mode=render_mode)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def get_env_spec(env_name: str, env: gym.Env) -> dict:
    """
    LLM 프롬프트용 환경 요약을 반환 (관측/액션 차원, 범위, 기본 안전 한계)
    """
    obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, "shape") else None
    act_dim = env.action_space.shape[0] if hasattr(env.action_space, "shape") else None
    act_low = env.action_space.low.tolist() if hasattr(env.action_space, "low") else None
    act_high = env.action_space.high.tolist() if hasattr(env.action_space, "high") else None

    spec = {
        "env_name": env_name,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "act_range": {"low": act_low, "high": act_high},
        "max_stage": 3,
    }

    # 환경별 변경 가능 파라미터 범위를 명시
    if "HalfCheetah" in env_name:
        spec.update(
            {
                "friction_range": [0.4, 1.5],
                "mass_scale_range": [0.7, 1.3],
                "slope_deg_range": [-5, 5],
            }
        )
    if "LunarLander" in env_name:
        spec.update(
            {
                "wind_noise_range": [0.0, 0.3],
                "gravity_scale_range": [0.8, 1.2],
            }
        )
    return spec