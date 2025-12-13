#!/usr/bin/env python3
"""
모든 저장된 모델에 대해 정상 및 강건성 평가를 수행하고 결과를 CSV로 저장하는 스크립트.
outputs/ 폴더 구조를 순회하며 각 모델을 찾고, wandb에서 config를 매칭하여 종합 평가 수행.
"""

import os
import sys
import csv
import yaml
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC, TD3, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse

# 프로젝트 루트를 PYTHONPATH에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rl_modules.evaluate import evaluate_robustness


class ModelEvaluator:
    """모델 평가 및 강건성 테스트를 수행하는 클래스"""
    
    def __init__(self, output_csv: str = "model_evaluation_results.csv"):
        self.output_csv = output_csv
        self.project_root = Path(__file__).parent.parent
        self.outputs_dir = self.project_root / "outputs"
        self.wandb_dir = self.project_root / "wandb"
        
        # CSV 헤더 초기화 (파일이 없으면 생성)
        if not Path(self.output_csv).exists():
            self._init_csv()
    
    def _init_csv(self):
        """CSV 파일 초기화 (헤더 작성)"""
        headers = [
            "timestep",
            "env_name",
            "algorithm",
            "seed",
            "training_type",  # baseline / llm-acl / adr
            "normal_mean_reward",
            "robust_mean_reward",
            "obs_noise_0.1",
            "obs_noise_0.2",
            "action_noise_0.1",
            "action_noise_0.2",
            "obs_dropout_0.1",
            "combined_mild",
            "combined_strong",
            "evaluation_date",
        ]
        
        with open(self.output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        print(f"CSV 파일 초기화 완료: {self.output_csv}")
    
    def find_all_models(self) -> List[Path]:
        """outputs/ 폴더에서 모든 final_model.zip 파일 찾기"""
        model_files = []
        
        # outputs/{env_name}/{algorithm}/{timestamp}/final_model.zip 구조 탐색
        if not self.outputs_dir.exists():
            print(f"경고: outputs 폴더를 찾을 수 없습니다: {self.outputs_dir}")
            return []

        for env_dir in self.outputs_dir.iterdir():
            if not env_dir.is_dir() or env_dir.name.startswith('.'):
                continue
            
            for algo_dir in env_dir.iterdir():
                if not algo_dir.is_dir():
                    continue
                
                for timestamp_dir in algo_dir.iterdir():
                    if not timestamp_dir.is_dir():
                        continue
                    
                    model_file = timestamp_dir / "final_model.zip"
                    if model_file.exists():
                        model_files.append(model_file)
        
        return sorted(model_files)
    
    def find_wandb_config(self, model_path: Path) -> Optional[Dict]:
        """
        모델 경로에 대응하는 wandb config 찾기.
        timestamp를 기준으로 wandb/run-{timestamp}-{id}/files/config.yaml 매칭.
        """
        # timestamp 추출: outputs/{env}/{algo}/{timestamp}/final_model.zip
        timestamp_str = model_path.parent.name  # 예: "20251205_120000"
        
        if not self.wandb_dir.exists():
            return None

        # wandb 폴더에서 해당 timestamp와 일치하는 run 찾기
        for run_dir in self.wandb_dir.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith("run-"):
                continue
            
            # run-20251205_120000-abc123def 형식에서 timestamp 부분 추출
            run_name = run_dir.name  # "run-20251205_120000-abc123def"
            if timestamp_str in run_name:
                config_file = run_dir / "files" / "config.yaml"
                if config_file.exists():
                    with open(config_file, 'r') as f:
                        config = yaml.safe_load(f)
                    return config
        
        return None
    
    def load_model(self, model_path: Path, algorithm: str):
        """알고리즘에 맞는 모델 로드"""
        algo_map = {
            "sac": SAC,
            "td3": TD3,
            "ppo": PPO,
            "a2c": A2C,
        }
        
        if algorithm.lower() not in algo_map:
            raise ValueError(f"지원하지 않는 알고리즘: {algorithm}")
        
        model_class = algo_map[algorithm.lower()]
        return model_class.load(str(model_path))
    
    def evaluate_normal(self, model, env_name: str, n_episodes: int = 5) -> Tuple[float, float]:
        """정상 환경에서 모델 평가"""
        env = gym.make(env_name)
        env = DummyVecEnv([lambda: env])
        
        episode_rewards = []
        
        for _ in range(n_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]
            
            episode_rewards.append(episode_reward)
        
        env.close()
        
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        return mean_reward, std_reward
    
    def evaluate_robust(self, model, env_name: str, n_episodes: int = 5) -> Dict[str, float]:
        """강건성 테스트 (rl_modules.evaluate.evaluate_robustness 사용)"""
        # rl_modules.evaluate.evaluate_robustness 호출
        # returns dict: {scenario_name: [rewards...]}
        raw_results = evaluate_robustness(model, env_name, episodes=n_episodes)
        
        results = {}
        all_means = []
        
        # 각 시나리오별 평균 계산
        for scenario, rewards in raw_results.items():
            mean_reward = np.mean(rewards)
            results[scenario] = mean_reward
            all_means.append(mean_reward)
            
        # 전체 강건성 평균 및 표준편차
        results['mean'] = np.mean(all_means)
        results['std'] = np.std(all_means)
        
        return results
    
    def _run_episodes(self, model, env, n_episodes: int) -> List[float]:
        """주어진 환경에서 n_episodes 실행하고 보상 리스트 반환"""
        episode_rewards = []
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
        
        return episode_rewards

    def _get_param_value(self, config: Dict, key: str, default=None):
        """wandb config에서 파라미터 값 추출 (value 키 처리)"""
        if key not in config:
            return default
        
        val = config[key]
        # wandb config는 보통 {'value': actual_value, 'desc': ...} 형태
        if isinstance(val, dict) and 'value' in val:
            return val['value']
        return val

    def extract_training_type(self, config: Optional[Dict]) -> str:
        """config에서 training_type 추출 (baseline / llm-acl / adr)"""
        if config is None:
            return ""
        
        # llm_enabled 확인
        llm_enabled = self._get_param_value(config, "llm_enabled", False)
        if isinstance(llm_enabled, str):
            llm_enabled = llm_enabled.lower() == "true"
            
        # adr.enabled 확인
        adr_config = self._get_param_value(config, "adr", {})
        adr_enabled = False
        
        if isinstance(adr_config, dict):
            # adr이 딕셔너리인 경우 enabled 키 확인
            if "enabled" in adr_config:
                # adr['enabled']가 {'value': true} 형태일 수 있음
                val = adr_config["enabled"]
                if isinstance(val, dict) and "value" in val:
                    adr_enabled = val["value"]
                else:
                    adr_enabled = val
        
        if isinstance(adr_enabled, str):
            adr_enabled = adr_enabled.lower() == "true"

        if llm_enabled:
            return "llm-acl"
        elif adr_enabled:
            return "adr"
        
        # 기본값은 baseline
        return "baseline"
    
    def evaluate_model_and_save(self, model_path: Path):
        """단일 모델 평가 및 CSV에 결과 추가"""
        print(f"\n{'='*80}")
        print(f"평가 중: {model_path}")
        print(f"{'='*80}")
        
        # 경로에서 환경 이름과 알고리즘 추출
        # outputs/{env_name}/{algorithm}/{timestamp}/final_model.zip
        parts = model_path.parts
        env_name = parts[-4]
        algorithm = parts[-3]
        timestamp = parts[-2]
        
        print(f"환경: {env_name}, 알고리즘: {algorithm}, 타임스탬프: {timestamp}")
        
        # wandb config 찾기
        config = self.find_wandb_config(model_path)
        
        if config:
            print(f"✓ WandB config 발견")
            seed = self._get_param_value(config, "seed", "")
            training_type = self.extract_training_type(config)
        else:
            print(f"✗ WandB config 없음 (빈 값으로 처리)")
            seed = ""
            training_type = ""
        
        try:
            # 모델 로드
            print(f"모델 로딩 중...")
            model = self.load_model(model_path, algorithm)
            
            # 정상 평가
            print(f"정상 환경 평가 중...")
            normal_mean, _ = self.evaluate_normal(model, env_name, n_episodes=5)
            print(f"  정상 평균 보상: {normal_mean:.2f}")
            
            # 강건성 평가 (1 에피소드)
            print(f"강건성 테스트 중 (1 에피소드)...")
            robust_results = self.evaluate_robust(model, env_name, n_episodes=1)
            print(f"  강건성 평균 보상: {robust_results['mean']:.2f}")
            
            # CSV에 결과 추가
            row = [
                timestamp,
                env_name,
                algorithm,
                seed,
                training_type,
                f"{normal_mean:.4f}",
                f"{robust_results['mean']:.4f}",
                f"{robust_results.get('obs_noise_0.1', 0):.4f}",
                f"{robust_results.get('obs_noise_0.2', 0):.4f}",
                f"{robust_results.get('action_noise_0.1', 0):.4f}",
                f"{robust_results.get('action_noise_0.2', 0):.4f}",
                f"{robust_results.get('obs_dropout_0.1', 0):.4f}",
                f"{robust_results.get('combined_mild', 0):.4f}",
                f"{robust_results.get('combined_strong', 0):.4f}",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ]
            
            with open(self.output_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            
            print(f"✓ 결과 저장 완료: {self.output_csv}")
            
        except Exception as e:
            print(f"✗ 평가 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def summarize_results(self):
        """평가 결과 요약 (환경/타입/알고리즘 별 평균 및 표준편차 계산)"""
        print(f"\n{'='*80}")
        print(f"결과 요약 집계 중...")
        print(f"{'='*80}")
        
        if not Path(self.output_csv).exists():
            print("결과 파일이 없습니다.")
            return

        # 데이터 로드
        data = {}  # (env, type, algo) -> {'normal': [], 'robust': []}
        
        with open(self.output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # training_type이 없는 경우 'unknown' 처리
                t_type = row.get('training_type', '')
                if not t_type: t_type = 'baseline' # 빈 문자열인 경우 baseline으로 취급
                
                key = (row['env_name'], t_type, row['algorithm'])
                
                if key not in data:
                    data[key] = {'normal': [], 'robust': []}
                
                try:
                    data[key]['normal'].append(float(row['normal_mean_reward']))
                    data[key]['robust'].append(float(row['robust_mean_reward']))
                except ValueError:
                    continue
        
        # 요약 결과 저장
        summary_csv = self.output_csv.replace('.csv', '_summary.csv')
        headers = [
            "env_name", "training_type", "algorithm", "count",
            "normal_reward_mean", "normal_reward_std",
            "robust_reward_mean", "robust_reward_std"
        ]
        
        print(f"\n{'='*80}")
        print(f"{'Environment':<25} | {'Type':<10} | {'Algo':<5} | {'N':<3} | {'Normal (Mean±Std)':<20} | {'Robust (Mean±Std)':<20}")
        print(f"{'-'*80}")
        
        with open(summary_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            
            # 키 정렬 (Env -> Type -> Algo)
            sorted_keys = sorted(data.keys())
            
            for key in sorted_keys:
                env, t_type, algo = key
                normal_scores = data[key]['normal']
                robust_scores = data[key]['robust']
                
                count = len(normal_scores)
                if count == 0:
                    continue
                    
                n_mean = np.mean(normal_scores)
                n_std = np.std(normal_scores)
                r_mean = np.mean(robust_scores)
                r_std = np.std(robust_scores)
                
                print(f"{env:<25} | {t_type:<10} | {algo:<5} | {count:<3} | {n_mean:>8.2f} ± {n_std:<8.2f} | {r_mean:>8.2f} ± {r_std:<8.2f}")
                
                writer.writerow([
                    env, t_type, algo, count,
                    f"{n_mean:.4f}", f"{n_std:.4f}",
                    f"{r_mean:.4f}", f"{r_std:.4f}"
                ])
                
        print(f"{'='*80}")
        print(f"요약 결과 저장 완료: {summary_csv}")

    def run_all(self):
        """모든 모델에 대해 평가 수행"""
        print(f"\n{'='*80}")
        print(f"모든 모델 평가 시작")
        print(f"{'='*80}\n")
        
        model_files = self.find_all_models()
        
        if not model_files:
            print("평가할 모델이 없습니다.")
            return
        
        print(f"발견된 모델 수: {len(model_files)}")
        
        for idx, model_path in enumerate(model_files, 1):
            print(f"\n[{idx}/{len(model_files)}] 진행 중...")
            self.evaluate_model_and_save(model_path)
        
        # 모든 평가가 끝난 후 요약 실행
        self.summarize_results()
        
        print(f"\n{'='*80}")
        print(f"모든 평가 완료!")
        print(f"결과 파일: {self.output_csv}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="outputs/ 폴더의 모든 모델을 평가하고 결과를 CSV로 저장"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_evaluation_results.csv",
        help="결과를 저장할 CSV 파일 경로 (기본: model_evaluation_results.csv)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="각 평가당 에피소드 수 (기본: 5)",
    )
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(output_csv=args.output)
    evaluator.run_all()


if __name__ == "__main__":
    main()
