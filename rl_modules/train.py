"""
RL 에이전트 학습 스크립트
Hydra를 사용하여 설정을 불러오고, 지정된 알고리즘으로 모델을 학습시킵니다.
"""

import os
import datetime
import hydra
from omegaconf import DictConfig
from stable_baselines3 import SAC, PPO, TD3, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import torch
import wandb

from rl_modules.utils import make_env
from rl_modules.custom_wrappers import LLMGuidedRobustnessWrapper
from rl_modules.custom_callbacks import LLMControlCallback, WandbLoggingCallback, ProgressCallback
from llm_core import LLMDecider


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """
    RL 에이전트를 학습시키는 메인 함수

    Args:
        cfg (DictConfig): Hydra 설정 객체
    """
    # wandb API 키 로드 및 로그인 # 여기는 본인 설정 필요
    wandb_api_key_path = os.path.join(os.path.dirname(__file__), "..", "wandb_info", "wandb_api_key")
    with open(wandb_api_key_path, "r") as f:
        wandb_api_key = f.read().strip()
    wandb.login(key=wandb_api_key)

    # 실험 디렉토리 생성 (환경/알고리즘/타임스탬프 구조)
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(outputs_dir, cfg.env.env_name, cfg.agent.algorithm, timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    # wandb 초기화 # 여기는 본인 설정 필요
    run_name = f"{cfg.agent.algorithm}_{cfg.env.env_name}_{'llm' if cfg.llm_enabled else 'baseline'}"
    wandb.init(
        entity="tatalintelli-university-of-seoul",
        project="dl-project-llm-robuster",
        name=run_name,
        tags=[cfg.agent.algorithm, cfg.env.env_name, "llm" if cfg.llm_enabled else "baseline"],
        config=dict(cfg)
    )

    # 환경 생성
    env = make_env(
        env_name=cfg.env.env_name,
        render_mode=cfg.env.render_mode,
        max_episode_steps=cfg.env.max_episode_steps
    )

    # 액션 차원 추출
    if hasattr(env.action_space, 'shape') and env.action_space.shape:
        action_dim = env.action_space.shape[0]
    elif hasattr(env.action_space, 'n'):
        action_dim = env.action_space.n  # Discrete action space
    else:
        raise ValueError(f"지원하지 않는 action space 타입: {type(env.action_space)}")

    # 학습 환경 설정 (LLM 활성화 여부에 따라)
    if cfg.llm_enabled:
        # LLM 가이드 Robustness 래퍼 적용
        wrapper = LLMGuidedRobustnessWrapper(env, action_dim)
        train_env = wrapper

        # LLM 의사결정 객체 생성
        llm_decider = LLMDecider(
            model_name=cfg.llm.model_name,
            use_4bit=cfg.llm.use_4bit,
            use_8bit=cfg.llm.use_8bit,
            mock=cfg.llm.mock
        )

        # LLM 제어 콜백 생성
        llm_callback = LLMControlCallback(
            wrapper=wrapper,
            llm_decider=llm_decider,
            update_freq=cfg.eval_freq,  # 평가 빈도와 동일하게 설정
            verbose=1
        )
    else:
        # 베이스라인 모드: 래퍼 없이 기본 환경 사용
        train_env = env
        llm_callback = None

    # 체크포인트 콜백 생성 (주기적 모델 저장)
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.checkpoint_freq,  # 체크포인트 저장 빈도
        save_path=exp_dir,
        name_prefix="checkpoint",
        verbose=1
    )

    # Wandb 로깅 콜백 생성
    wandb_logging_callback = WandbLoggingCallback(
        log_freq=cfg.log_interval,  # 로그 간격
        verbose=1
    )

    # 진행 상황 콜백 생성
    progress_callback = ProgressCallback(
        total_timesteps=cfg.total_timesteps,
        update_freq=cfg.log_interval,  # 로그 간격과 동일
        verbose=1
    )

    # 콜백 리스트 구성
    callbacks = [checkpoint_callback, wandb_logging_callback, progress_callback]
    if cfg.llm_enabled:
        callbacks.append(llm_callback)
    callback_list = CallbackList(callbacks)

    # 알고리즘 클래스 매핑 (확장성을 위한 딕셔너리 사용)
    algorithm_map = {
        "sac": SAC,
        "ppo": PPO,
        "td3": TD3,
        "a2c": A2C
    }

    # 선택된 알고리즘 확인
    algorithm_name = cfg.agent.algorithm
    if algorithm_name not in algorithm_map:
        raise ValueError(f"지원하지 않는 알고리즘입니다: {algorithm_name}. "
                        f"지원되는 알고리즘: {list(algorithm_map.keys())}")

    # 모델 클래스 가져오기
    agent_class = algorithm_map[algorithm_name]

    # 설정에서 algorithm 필드를 제외하고 모델 초기화에 사용
    model_kwargs = {k: v for k, v in cfg.agent.items() if k != "algorithm"}
    model_kwargs["device"] = cfg.device  # 디바이스 설정 추가
    model = agent_class("MlpPolicy", train_env, **model_kwargs)

    # 학습 실행 (콜백 적용)
    mode_desc = "LLM 가이드 Robustness" if cfg.llm_enabled else "베이스라인"
    print(f"{algorithm_name.upper()} 알고리즘으로 학습을 시작합니다...")
    print(f"{mode_desc} 모드가 활성화되었습니다.")
    model.learn(total_timesteps=cfg.total_timesteps, callback=callback_list)
    print("학습이 완료되었습니다.")

    # 최종 모델 저장
    final_model_path = os.path.join(exp_dir, "final_model")
    model.save(final_model_path)
    print(f"최종 모델이 {final_model_path}에 저장되었습니다.")

    # wandb 종료
    wandb.finish()


if __name__ == "__main__":
    train()