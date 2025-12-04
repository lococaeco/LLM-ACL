"""
모델 평가 및 강건성 테스트 스크립트
학습된 RL 모델을 평가하고, 다양한 교란 시나리오에서 강건성을 측정합니다.
"""

import os
import argparse
import numpy as np
from stable_baselines3 import SAC, PPO, TD3, A2C
from gymnasium.wrappers import RecordVideo

from rl_modules.utils import make_env


def evaluate_model(
    model_path: str, 
    env_name: str, 
    episodes: int = 5, 
    render_video: bool = False, 
    video_folder: str = None,
    test_robustness: bool = False,
):
    """
    학습된 모델을 평가하고 선택적으로 강건성 테스트를 수행합니다.
    """
    # 모델 파일에서 알고리즘 추론
    model_path_lower = model_path.lower()
    if "sac" in model_path_lower:
        algorithm = "sac"
        model_class = SAC
    elif "ppo" in model_path_lower:
        algorithm = "ppo"
        model_class = PPO
    elif "td3" in model_path_lower:
        algorithm = "td3"
        model_class = TD3
    elif "a2c" in model_path_lower:
        algorithm = "a2c"
        model_class = A2C
    else:
        raise ValueError(f"모델 경로에서 알고리즘을 추론할 수 없습니다: {model_path}")

    print(f"모델 로드 중: {model_path}")
    model = model_class.load(model_path)

    # 환경 생성
    env = make_env(env_name, render_mode="rgb_array" if render_video else None)

    # 비디오 녹화 설정
    if render_video:
        if video_folder is None:
            model_dir = os.path.dirname(model_path)
            video_folder = os.path.join(model_dir, "videos")
        os.makedirs(video_folder, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda episode_id: True,
            name_prefix=f"{algorithm}_{env_name}"
        )
        print(f"비디오가 {video_folder}에 저장됩니다.")

    # 기본 평가
    print(f"\n{'='*60}")
    print(f"{algorithm.upper()} 모델 평가 ({episodes} 에피소드)")
    print(f"{'='*60}")
    
    normal_rewards, normal_lengths = _run_evaluation(model, env, episodes)
    
    print(f"\n[정상 환경 결과]")
    print(f"  평균 보상: {np.mean(normal_rewards):.2f} ± {np.std(normal_rewards):.2f}")
    print(f"  평균 길이: {np.mean(normal_lengths):.1f}")

    env.close()

    # 강건성 테스트
    if test_robustness:
        print(f"\n{'='*60}")
        print("강건성 테스트")
        print(f"{'='*60}")
        
        robustness_results = evaluate_robustness(model, env_name, episodes)
        
        print(f"\n[강건성 결과 요약]")
        print(f"  정상 환경: {np.mean(normal_rewards):.2f}")
        for scenario, rewards in robustness_results.items():
            ratio = np.mean(rewards) / np.mean(normal_rewards) if np.mean(normal_rewards) > 0 else 0
            print(f"  {scenario}: {np.mean(rewards):.2f} (성능 비율: {ratio:.2%})")


def evaluate_robustness(model, env_name: str, episodes: int = 5) -> dict:
    """다양한 교란 시나리오에서 모델 평가"""
    
    perturbation_scenarios = {
        "obs_noise_0.1": {"type": "obs_noise", "value": 0.1},
        "obs_noise_0.2": {"type": "obs_noise", "value": 0.2},
        "action_noise_0.1": {"type": "action_noise", "value": 0.1},
        "action_noise_0.2": {"type": "action_noise", "value": 0.2},
        "obs_dropout_0.1": {"type": "obs_dropout", "value": 0.1},
        "combined_mild": {"type": "combined", "obs_noise": 0.05, "action_noise": 0.05},
        "combined_strong": {"type": "combined", "obs_noise": 0.15, "action_noise": 0.1},
    }
    
    results = {}
    
    for scenario_name, scenario in perturbation_scenarios.items():
        print(f"\n테스트 중: {scenario_name}")
        env = make_env(env_name, None)
        rewards = []
        
        for ep in range(episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # 관측 교란
                if scenario["type"] == "obs_noise":
                    obs = obs + np.random.normal(0, scenario["value"], obs.shape)
                elif scenario["type"] == "obs_dropout":
                    mask = np.random.random(obs.shape) > scenario["value"]
                    obs = obs * mask
                elif scenario["type"] == "combined":
                    obs = obs + np.random.normal(0, scenario["obs_noise"], obs.shape)
                
                action, _ = model.predict(obs, deterministic=True)
                
                # 행동 교란
                if scenario["type"] == "action_noise":
                    action = action + np.random.normal(0, scenario["value"], action.shape)
                elif scenario["type"] == "combined":
                    action = action + np.random.normal(0, scenario["action_noise"], action.shape)
                
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            rewards.append(episode_reward)
            print(f"  에피소드 {ep+1}: {episode_reward:.2f}")
        
        env.close()
        results[scenario_name] = rewards
    
    return results


def _run_evaluation(model, env, episodes: int):
    """기본 평가 실행"""
    rewards = []
    lengths = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
        print(f"  에피소드 {ep+1}: 보상={episode_reward:.2f}, 길이={episode_length}")
    
    return rewards, lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL 모델 평가 및 강건성 테스트")
    parser.add_argument("--model_path", type=str, required=True, help="모델 파일 경로")
    parser.add_argument("--env_name", type=str, required=True, help="환경 이름")
    parser.add_argument("--episodes", type=int, default=5, help="평가 에피소드 수")
    parser.add_argument("--render_video", action="store_true", help="비디오 녹화 여부")
    parser.add_argument("--video_folder", type=str, default=None, help="비디오 저장 폴더")
    parser.add_argument("--test_robustness", action="store_true", help="강건성 테스트 수행")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        env_name=args.env_name,
        episodes=args.episodes,
        render_video=args.render_video,
        video_folder=args.video_folder,
        test_robustness=args.test_robustness,
    )