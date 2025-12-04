"""
ALRT (Adaptive LLM-Guided Robustness Training) 학습 스크립트
Hydra 설정을 불러와 적응형 LLM 커리큘럼 기반 강건 학습을 수행.
"""

import os
import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC, PPO, TD3, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import wandb

from rl_modules.utils import make_env, get_env_spec, get_target_reward
from rl_modules.custom_wrappers import AdaptiveCurriculumWrapper, ALRTEnvWrapper
from rl_modules.custom_callbacks import ALRTCallback, WandbLoggingCallback, ProgressCallback
from llm_core.decider import LLMDecider


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    """Hydra 진입점"""
    # wandb 로그인
    mingu = "0fccac088b1041eeae51eaad8941a490bf71592c"
    wandb.login(key=mingu)

    # 실험 디렉토리 생성
    outputs_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(outputs_dir, cfg.env.env_name, cfg.agent.algorithm, timestamp)
    os.makedirs(exp_dir, exist_ok=True)

    # 실험 모드 결정
    if cfg.llm_enabled:
        mode = "ALRT" if not cfg.llm.mock else "ALRT-mock"
    else:
        mode = "baseline"

    run_name = f"{cfg.agent.algorithm}_{cfg.env.env_name}_{mode}"
    wandb.init(
        entity="tatalintelli-university-of-seoul",
        project="dl-project-llm-robuster",
        name=run_name,
        tags=[cfg.agent.algorithm, cfg.env.env_name, mode],
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # 환경 생성 및 스펙 추출
    print(f"[DEBUG] PID {os.getpid()}: 환경 생성 시작 ({cfg.env.env_name})", flush=True)
    base_env = make_env(cfg.env.env_name, cfg.env.render_mode, cfg.env.max_episode_steps)
    env_spec = get_env_spec(cfg.env.env_name, base_env)

    # 평가용 환경 생성 함수 (강건성 테스트용)
    def make_eval_env():
        return make_env(cfg.env.env_name, None, cfg.env.max_episode_steps)

    # 목표 보상 설정
    target_reward = get_target_reward(cfg.env.env_name, cfg.curriculum)

    # 적응형 커리큘럼 래퍼 적용
    if cfg.curriculum.enabled:
        train_env = ALRTEnvWrapper(
            env=base_env,
            max_episode_steps=cfg.env.max_episode_steps,
            safety_clip=dict(cfg.curriculum.safety_clip),
        )
    elif cfg.get("adr", {}).get("enabled", False):
        print(f"[ADR] Standard ADR Enabled (Initial Range: {cfg.adr.initial_range})")
        from rl_modules.custom_wrappers import StandardADRWrapper
        train_env = StandardADRWrapper(
            env=base_env,
            initial_range=cfg.adr.initial_range,
            expand_rate=cfg.adr.expand_rate
        )
    else:
        train_env = base_env

    # LLM 디사이더 준비
    print(f"[DEBUG] PID {os.getpid()}: LLMDecider 초기화 시작 (model={cfg.llm.model_name})", flush=True)
    llm_decider = LLMDecider(
        model_name=cfg.llm.model_name,
        use_4bit=cfg.llm.use_4bit,
        use_8bit=cfg.llm.use_8bit,
        mock=(not cfg.llm_enabled) or cfg.llm.mock,
        prompt_cfg=dict(cfg.llm),
    )
    print(f"[DEBUG] PID {os.getpid()}: LLMDecider 초기화 완료", flush=True)

    # ALRT 콜백
    alrt_callback = None
    adr_callback = None

    if cfg.curriculum.enabled:
        alrt_callback = ALRTCallback(
            wrapper=train_env,
            llm_decider=llm_decider,
            curriculum_cfg=cfg.curriculum,
            env_spec=env_spec,
            base_env_fn=make_eval_env,
            target_reward=target_reward,
            total_timesteps=cfg.total_timesteps,
            use_llm=cfg.llm_enabled and not cfg.llm.mock,
            verbose=1,
        )
    elif cfg.get("adr", {}).get("enabled", False):
        from rl_modules.custom_callbacks import StandardADRCallback
        adr_callback = StandardADRCallback(
            env=train_env,
            update_freq=cfg.adr.update_freq,
            threshold=cfg.adr.threshold,
            target_reward=target_reward,
            verbose=1
        )

    # 체크포인트/로깅 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.checkpoint_freq,
        save_path=exp_dir,
        name_prefix="checkpoint",
        verbose=1,
    )
    wandb_logging_callback = WandbLoggingCallback(log_freq=cfg.log_interval, verbose=1)
    progress_callback = ProgressCallback(
        total_timesteps=cfg.total_timesteps, 
        update_freq=cfg.log_interval, 
        verbose=1
    )

    callbacks = [checkpoint_callback, wandb_logging_callback, progress_callback]
    if alrt_callback:
        callbacks.append(alrt_callback)
    if adr_callback:
        callbacks.append(adr_callback)
    callback_list = CallbackList(callbacks)

    # 알고리즘 선택
    algorithm_map = {"sac": SAC, "ppo": PPO, "td3": TD3, "a2c": A2C}
    algorithm_name = cfg.agent.algorithm
    if algorithm_name not in algorithm_map:
        raise ValueError(f"지원하지 않는 알고리즘: {algorithm_name}")
    agent_class = algorithm_map[algorithm_name]

    # 모델 초기화
    print(f"[DEBUG] PID {os.getpid()}: RL 에이전트 모델 초기화 시작 ({algorithm_name})", flush=True)
    model_kwargs = {k: v for k, v in cfg.agent.items() if k != "algorithm"}
    model_kwargs["device"] = cfg.device
    model = agent_class("MlpPolicy", train_env, **model_kwargs)

    print(f"\n{'='*60}")
    print(f"[ALRT] 학습 시작")
    print(f"  - 알고리즘: {algorithm_name.upper()}")
    print(f"  - 환경: {cfg.env.env_name}")
    print(f"  - 모드: {mode}")
    print(f"  - 목표 보상: {target_reward}")
    print(f"  - 총 스텝: {cfg.total_timesteps:,}")
    print(f"{'='*60}\n")

    try:
        print(f"[DEBUG] PID {os.getpid()}: 학습 루프 시작 (model.learn 호출)", flush=True)
        model.learn(total_timesteps=cfg.total_timesteps, callback=callback_list)
        print("\n학습이 완료되었습니다.")
        
        # 최종 모델 저장
        final_model_path = os.path.join(exp_dir, "final_model")
        model.save(final_model_path)
        print(f"최종 모델 저장 완료: {final_model_path}")
        
        # 최종 강건성 평가
        if alrt_callback:
            print("\n[ALRT] 최종 강건성 평가 중...")
            alrt_callback._evaluate_robustness()
            print(f"최종 강건성 점수: {alrt_callback.last_robustness_score:.2f}")
            
    except Exception as e:
        print(f"학습 중 오류 발생: {e}")
        raise e
    finally:
        wandb.finish()


if __name__ == "__main__":
    train()