"""
RL 에이전트 학습 스크립트
Hydra 설정을 불러와 LLM 커리큘럼 기반 강건 학습을 수행.
"""

import os
import datetime
import hydra
from omegaconf import DictConfig
from stable_baselines3 import SAC, PPO, TD3, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import wandb

from rl_modules.utils import make_env, get_env_spec
from rl_modules.custom_wrappers import CurriculumEnvWrapper
from rl_modules.custom_callbacks import LLMCurriculumCallback, WandbLoggingCallback, ProgressCallback
from llm_core.decider import LLMDecider


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Hydra 진입점"""
    # wandb 로그인 (보안을 위해 key를 환경변수로 관리 권장)
    sangwoo = "7c6164675504dd1412328d937442bc75fb380454"
    wandb.login(key=sangwoo)

    # 실험 디렉토리 생성
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(outputs_dir, cfg.env.env_name, cfg.agent.algorithm, timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    run_name = f"{cfg.agent.algorithm}_{cfg.env.env_name}_{'curriculum' if cfg.llm_enabled else 'baseline'}"
    wandb.init(
        entity="sangwoo6999-unist",
        project="dl-project-llm-robuster",
        name=run_name,
        tags=[cfg.agent.algorithm, cfg.env.env_name, "curriculum" if cfg.llm_enabled else "baseline"],
        config=dict(cfg),
    )

    # 환경 생성 및 스펙 추출
    base_env = make_env(cfg.env.env_name, cfg.env.render_mode, cfg.env.max_episode_steps)
    env_spec = get_env_spec(cfg.env.env_name, base_env)

    # 커리큘럼 래퍼 적용
    if cfg.curriculum.enabled:
        train_env = CurriculumEnvWrapper(
            env=base_env,
            max_episode_steps=cfg.env.max_episode_steps,
            safety_clip=dict(cfg.curriculum.safety_clip),
        )
    else:
        train_env = base_env

    # LLM 디사이더 준비 (llm_enabled가 False이거나 mock=True면 mock 플랜 사용)
    llm_decider = LLMDecider(
        model_name=cfg.llm.model_name,
        use_4bit=cfg.llm.use_4bit,
        use_8bit=cfg.llm.use_8bit,
        mock=(not cfg.llm_enabled) or cfg.llm.mock,
        prompt_cfg=dict(cfg.llm),
    )

    # 커리큘럼 콜백
    curriculum_callback = None
    if cfg.curriculum.enabled:
        curriculum_callback = LLMCurriculumCallback(
            wrapper=train_env,
            llm_decider=llm_decider,
            curriculum_cfg=cfg.curriculum,
            env_spec=env_spec,
            use_llm=cfg.llm_enabled and not cfg.llm.mock,
            verbose=1,
        )

    # 체크포인트/로깅 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.checkpoint_freq,
        save_path=exp_dir,
        name_prefix="checkpoint",
        verbose=1,
    )
    wandb_logging_callback = WandbLoggingCallback(log_freq=cfg.log_interval, verbose=1)
    progress_callback = ProgressCallback(total_timesteps=cfg.total_timesteps, update_freq=cfg.log_interval, verbose=1)

    callbacks = [checkpoint_callback, wandb_logging_callback, progress_callback]
    if curriculum_callback:
        callbacks.append(curriculum_callback)
    callback_list = CallbackList(callbacks)

    # 알고리즘 선택
    algorithm_map = {"sac": SAC, "ppo": PPO, "td3": TD3, "a2c": A2C}
    algorithm_name = cfg.agent.algorithm
    if algorithm_name not in algorithm_map:
        raise ValueError(f"지원하지 않는 알고리즘: {algorithm_name}")
    agent_class = algorithm_map[algorithm_name]

    # 모델 초기화
    model_kwargs = {k: v for k, v in cfg.agent.items() if k != "algorithm"}
    model_kwargs["device"] = cfg.device
    model = agent_class("MlpPolicy", train_env, **model_kwargs)

    mode_desc = "LLM 커리큘럼" if cfg.llm_enabled else "베이스라인"
    print(f"\n[{algorithm_name.upper()}] 학습 시작 | 모드: {mode_desc}")

    try:
        model.learn(total_timesteps=cfg.total_timesteps, callback=callback_list)
        print("학습이 완료되었습니다.")
        final_model_path = os.path.join(exp_dir, "final_model")
        model.save(final_model_path)
        print(f"최종 모델 저장 완료: {final_model_path}")
    except Exception as e:
        print(f"학습 중 오류 발생: {e}")
        raise e
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()