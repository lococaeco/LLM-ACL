"""
RL 환경 생성 및 유틸리티 함수들
"""

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from typing import Any, Dict, Optional
from omegaconf import DictConfig


def make_env(env_name: str, render_mode: str = None, max_episode_steps: int = None) -> gym.Env:
    """
    지정된 환경을 생성하고 필요한 래퍼를 적용하는 함수
    """
    env = gym.make(env_name, render_mode=render_mode)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env


def get_env_spec(env_name: str, env: gym.Env) -> Dict[str, Any]:
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

    # 환경별 특성 및 목표 보상
    env_configs = {
        "HalfCheetah-v4": {
            "target_reward": 8000,
            "friction_range": [0.4, 1.5],
            "mass_scale_range": [0.7, 1.3],
            "description": "2D 치타 로봇이 앞으로 빠르게 달리는 환경",
        },
        "Hopper-v4": {
            "target_reward": 3000,
            "friction_range": [0.5, 1.5],
            "mass_scale_range": [0.8, 1.2],
            "description": "한 발로 뛰는 로봇이 균형을 유지하며 전진",
        },
        "Walker2d-v4": {
            "target_reward": 5000,
            "friction_range": [0.5, 1.5],
            "mass_scale_range": [0.8, 1.2],
            "description": "2족 보행 로봇이 걸어서 전진",
        },
        "Ant-v4": {
            "target_reward": 6000,
            "friction_range": [0.4, 1.5],
            "mass_scale_range": [0.7, 1.3],
            "description": "4족 개미 로봇이 전진",
        },
        "Humanoid-v4": {
            "target_reward": 6000,
            "friction_range": [0.6, 1.4],
            "mass_scale_range": [0.8, 1.2],
            "description": "인간형 로봇이 균형을 유지하며 걷기",
        },
        "LunarLanderContinuous-v3": {
            "target_reward": 250,
            "wind_noise_range": [0.0, 0.3],
            "gravity_scale_range": [0.8, 1.2],
            "description": "달 착륙선을 안전하게 착륙시키는 환경",
        },
    }

    # 환경별 설정 적용
    for env_key, config in env_configs.items():
        if env_key in env_name:
            spec.update(config)
            break

    return spec


def get_target_reward(env_name: str, curriculum_cfg: Optional[DictConfig] = None) -> float:
    """환경별 목표 보상 반환"""
    
    # 기본 목표 보상
    default_targets = {
        "HalfCheetah-v4": 8000,
        "Hopper-v4": 3000,
        "Walker2d-v4": 5000,
        "Ant-v4": 6000,
        "Humanoid-v4": 6000,
        "LunarLanderContinuous-v3": 250,
    }
    
    # curriculum config에서 먼저 찾기
    if curriculum_cfg is not None:
        alrt_cfg = curriculum_cfg.get("alrt", {})
        target_rewards = alrt_cfg.get("target_rewards", {})
        for env_key, reward in target_rewards.items():
            if env_key in env_name:
                return float(reward)
    
    # 기본값에서 찾기
    for env_key, reward in default_targets.items():
        if env_key in env_name:
            return float(reward)
    
    # 못 찾으면 기본값
    return 3000.0


def calculate_perturbation_intensity(stage: int, max_stage: int, base_intensity: float = 1.0) -> float:
    """
    현재 stage에 따른 교란 강도 계산
    stage가 높을수록 더 강한 교란
    """
    if max_stage <= 0:
        return base_intensity
    ratio = stage / max_stage
    return base_intensity * (0.3 + 0.7 * ratio)  # 30% ~ 100% 범위