#!/bin/bash
set -euo pipefail

# 베이스라인 RL 학습 실행 스크립트
# LLM 기능을 끄고 기본 환경에서 각 알고리즘을 학습시킵니다.

# Python 경로 설정
export PYTHONPATH="/home/mlic/mingukang/DL_Project:$PYTHONPATH"

# GPU 설정 (필요시 변경: 0, 1, 또는 all)
export CUDA_VISIBLE_DEVICES=0,1

echo "=== 베이스라인 RL 학습 시작 ==="

# MuJoCo 환경 (HalfCheetah-v4)에서 각 알고리즘 학습
# echo "1. SAC 알고리즘 - MuJoCo (HalfCheetah-v4)"
# python rl_modules/train.py agent=sac env=mujoco llm_enabled=false

# echo "2. PPO 알고리즘 - MuJoCo (HalfCheetah-v4)"
# python rl_modules/train.py agent=ppo env=mujoco llm_enabled=false

# echo "3. TD3 알고리즘 - MuJoCo (HalfCheetah-v4)"
# python rl_modules/train.py agent=td3 env=mujoco llm_enabled=false

# echo "4. A2C 알고리즘 - MuJoCo (HalfCheetah-v4)"
# python rl_modules/train.py agent=a2c env=mujoco llm_enabled=false

# LunarLander 환경에서 각 알고리즘 학습
echo "5. SAC 알고리즘 - LunarLander-v2"
python rl_modules/train.py agent=sac env=lunarlander llm_enabled=false

# echo "6. PPO 알고리즘 - LunarLander-v2"
# python rl_modules/train.py agent=ppo env=lunarlander llm_enabled=false

# echo "7. TD3 알고리즘 - LunarLander-v2"
# python rl_modules/train.py agent=td3 env=lunarlander llm_enabled=false

# echo "8. A2C 알고리즘 - LunarLander-v2"
# python rl_modules/train.py agent=a2c env=lunarlander llm_enabled=false

echo "=== 모든 베이스라인 학습 완료 ==="