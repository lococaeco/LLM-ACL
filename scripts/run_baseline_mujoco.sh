#!/bin/bash
set -euo pipefail

# 베이스라인 RL 학습 실행 스크립트
# LLM 기능을 끄고 기본 환경에서 각 알고리즘을 학습시킵니다.

# Python 경로 설정
export PYTHONPATH="/home/mlic/mingukang/DL_Project:$PYTHONPATH"

# GPU 설정 (단일 GPU 사용 권장 - Stable-Baselines3은 다중 GPU를 기본 지원하지 않음)
export CUDA_VISIBLE_DEVICES=0

echo "=== 베이스라인 RL 학습 시작 ==="

# HalfCheetah-v4 환경에서 각 알고리즘 학습
# echo "=== HalfCheetah-v4 환경 학습 시작 ==="
# echo "SAC 알고리즘 - HalfCheetah-v4 (GPU 4)"
# CUDA_VISIBLE_DEVICES=4 python rl_modules/train.py agent=sac env=mujoco llm_enabled=false &

# echo "PPO 알고리즘 - HalfCheetah-v4 (GPU 5)"
# CUDA_VISIBLE_DEVICES=5 python rl_modules/train.py agent=ppo env=mujoco llm_enabled=false &

# echo "TD3 알고리즘 - HalfCheetah-v4 (GPU 6)"
# CUDA_VISIBLE_DEVICES=6 python rl_modules/train.py agent=td3 env=mujoco llm_enabled=false &

# echo "A2C 알고리즘 - HalfCheetah-v4 (GPU 7)"
# CUDA_VISIBLE_DEVICES=7 python rl_modules/train.py agent=a2c env=mujoco llm_enabled=false &

# # HalfCheetah 학습 완료 대기
# wait

# Hopper-v4 환경에서 각 알고리즘 병렬 학습
echo "=== Hopper-v4 환경 학습 시작 ==="
echo "SAC 알고리즘 - Hopper-v4 (GPU 4)"
CUDA_VISIBLE_DEVICES=4 python rl_modules/train.py agent=sac env=hopper llm_enabled=false &

echo "PPO 알고리즘 - Hopper-v4 (GPU 5)"
CUDA_VISIBLE_DEVICES=5 python rl_modules/train.py agent=ppo env=hopper llm_enabled=false &

echo "TD3 알고리즘 - Hopper-v4 (GPU 6)"
CUDA_VISIBLE_DEVICES=6 python rl_modules/train.py agent=td3 env=hopper llm_enabled=false &

echo "A2C 알고리즘 - Hopper-v4 (GPU 7)"
CUDA_VISIBLE_DEVICES=7 python rl_modules/train.py agent=a2c env=hopper llm_enabled=false &

# Hopper 학습 완료 대기
wait

# Ant-v4 환경에서 각 알고리즘 병렬 학습
echo "=== Ant-v4 환경 학습 시작 ==="
echo "SAC 알고리즘 - Ant-v4 (GPU 4)"
CUDA_VISIBLE_DEVICES=4 python rl_modules/train.py agent=sac env=ant llm_enabled=false &

echo "PPO 알고리즘 - Ant-v4 (GPU 5)"
CUDA_VISIBLE_DEVICES=5 python rl_modules/train.py agent=ppo env=ant llm_enabled=false &

echo "TD3 알고리즘 - Ant-v4 (GPU 6)"
CUDA_VISIBLE_DEVICES=6 python rl_modules/train.py agent=td3 env=ant llm_enabled=false &

echo "A2C 알고리즘 - Ant-v4 (GPU 7)"
CUDA_VISIBLE_DEVICES=7 python rl_modules/train.py agent=a2c env=ant llm_enabled=false &

# Ant 학습 완료 대기
wait

# Humanoid-v4 환경에서 각 알고리즘 병렬 학습
echo "=== Humanoid-v4 환경 학습 시작 ==="
echo "SAC 알고리즘 - Humanoid-v4 (GPU 4)"
CUDA_VISIBLE_DEVICES=4 python rl_modules/train.py agent=sac env=humanoid llm_enabled=false &

echo "PPO 알고리즘 - Humanoid-v4 (GPU 5)"
CUDA_VISIBLE_DEVICES=5 python rl_modules/train.py agent=ppo env=humanoid llm_enabled=false &

echo "TD3 알고리즘 - Humanoid-v4 (GPU 6)"
CUDA_VISIBLE_DEVICES=6 python rl_modules/train.py agent=td3 env=humanoid llm_enabled=false &

echo "A2C 알고리즘 - Humanoid-v4 (GPU 7)"
CUDA_VISIBLE_DEVICES=7 python rl_modules/train.py agent=a2c env=humanoid llm_enabled=false &

# Humanoid 학습 완료 대기
wait

echo "=== 모든 베이스라인 학습 완료 ==="