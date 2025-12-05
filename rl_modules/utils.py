"""
RL 환경 생성 및 유틸리티 함수들
"""

import gymnasium as gym
from gymnasium.wrappers import TimeLimit


def make_env(env_name: str, render_mode: str = None, max_episode_steps: int = None) -> gym.Env:
    """
    지정된 환경을 생성하고 필요한 래퍼를 적용하는 함수

    Args:
        env_name (str): 생성할 환경의 이름 (예: "LunarLander-v2", "HalfCheetah-v4")
        render_mode (str, optional): 렌더링 모드. 기본값은 None
        max_episode_steps (int, optional): 최대 에피소드 스텝 수. 기본값은 None

    Returns:
        gym.Env: 생성된 환경 객체
    """
    # 환경 생성
    env = gym.make(env_name, render_mode=render_mode)

    # 최대 에피소드 길이 제한 적용 (필요시)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)

    return env