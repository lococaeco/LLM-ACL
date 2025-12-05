"""
모델 평가 및 비디오 녹화 스크립트
학습된 RL 모델을 평가하고, 선택적으로 비디오로 녹화합니다.
"""

import os
import argparse
import numpy as np
from stable_baselines3 import SAC, PPO, TD3, A2C
from gymnasium.wrappers import RecordVideo

from .utils import make_env


def evaluate_model(model_path: str, env_name: str, episodes: int = 5, render_video: bool = False, video_folder: str = None):
    """
    학습된 모델을 평가하고 선택적으로 비디오를 녹화합니다.

    Args:
        model_path (str): 평가할 모델 파일 경로
        env_name (str): 환경 이름
        episodes (int): 평가할 에피소드 수
        render_video (bool): 비디오 녹화 여부
        video_folder (str): 비디오 저장 폴더 (None이면 모델 폴더에 저장)
    """
    # 모델 파일에서 알고리즘 추론 (파일명에 알고리즘 포함 가정)
    model_filename = os.path.basename(model_path)
    if "sac" in model_filename.lower():
        algorithm = "sac"
        model_class = SAC
    elif "ppo" in model_filename.lower():
        algorithm = "ppo"
        model_class = PPO
    elif "td3" in model_filename.lower():
        algorithm = "td3"
        model_class = TD3
    elif "a2c" in model_filename.lower():
        algorithm = "a2c"
        model_class = A2C
    else:
        raise ValueError(f"모델 파일명에서 알고리즘을 추론할 수 없습니다: {model_filename}")

    # 모델 로드
    print(f"모델 로드 중: {model_path}")
    model = model_class.load(model_path)

    # 환경 생성
    env = make_env(env_name, render_mode="rgb_array" if render_video else None)

    # 비디오 녹화 설정
    if render_video:
        if video_folder is None:
            # 모델 폴더에 videos 서브폴더 생성
            model_dir = os.path.dirname(model_path)
            video_folder = os.path.join(model_dir, "videos")

        os.makedirs(video_folder, exist_ok=True)

        # RecordVideo 래퍼 적용
        env = RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda episode_id: True,  # 모든 에피소드 녹화
            name_prefix=f"{algorithm}_{env_name}"
        )
        print(f"비디오가 {video_folder}에 저장됩니다.")

    # 평가 실행
    episode_rewards = []
    episode_lengths = []

    print(f"\n{algorithm.upper()} 모델 평가 시작 ({episodes} 에피소드)")
    print("=" * 50)

    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"에피소드 {episode + 1}: 보상 = {episode_reward:.2f}, 길이 = {episode_length}")

    env.close()

    # 결과 요약
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    print("\n" + "=" * 50)
    print("평가 결과 요약:")
    print(".2f")
    print(".2f")
    print(".1f")
    print("=" * 50)

    return mean_reward, std_reward, mean_length


def main():
    """메인 함수: 명령줄 인자 처리"""
    parser = argparse.ArgumentParser(description="RL 모델 평가 및 비디오 녹화")
    parser.add_argument("--model_path", type=str, required=True,
                       help="평가할 모델 파일 경로 (예: outputs/HalfCheetah-v4/sac/20251205_120000/final_model.zip)")
    parser.add_argument("--env_name", type=str, required=True,
                       help="환경 이름 (예: HalfCheetah-v4, LunarLander-v2)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="평가할 에피소드 수 (기본: 5)")
    parser.add_argument("--render_video", action="store_true",
                       help="비디오 녹화 활성화")
    parser.add_argument("--video_folder", type=str, default=None,
                       help="비디오 저장 폴더 (기본: 모델 폴더/videos)")

    args = parser.parse_args()

    # PYTHONPATH 설정 (스크립트 직접 실행 시)
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # 평가 실행
    evaluate_model(
        model_path=args.model_path,
        env_name=args.env_name,
        episodes=args.episodes,
        render_video=args.render_video,
        video_folder=args.video_folder
    )


if __name__ == "__main__":
    main()