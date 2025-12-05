#!/bin/bash
set -euo pipefail

# LLM 가이드 Robustness RL 학습 실행 스크립트
# LLM 기능을 켜고 Robustness 제어를 적용하여 학습시킵니다.

# Python 경로 설정
export PYTHONPATH="/home/mlic/mingukang/DL_Project:$PYTHONPATH"
export PYTHONPATH="/home/mlic/sangwoo/DL_Project:$PYTHONPATH"

# GPU 설정 (필요시 변경: 0, 1, 또는 all)
export CUDA_VISIBLE_DEVICES=0

echo "=== LLM 가이드 Robustness RL 학습 시작 ==="

# 모의 LLM 모드로 테스트 (실제 LLM 없이 랜덤 마스크)
echo "1. SAC + LLM (모의 모드) - MuJoCo (HalfCheetah-v4)"
python rl_modules/train.py agent=sac env=mujoco llm_enabled=true llm.mock=true

echo "2. PPO + LLM (모의 모드) - LunarLanderContinuous-v3"
python rl_modules/train.py agent=ppo env=lunarlander llm_enabled=true llm.mock=true

# 실제 LLM 사용 예시 (모델이 설치된 경우)
# echo "3. SAC + 실제 LLM - MuJoCo (HalfCheetah-v4)"
# python rl_modules/train.py agent=sac env=mujoco llm.mock=false llm.model_name="meta-llama/Meta-Llama-3-8B-Instruct"

# 4bit 양자화 사용 예시
# echo "4. PPO + 4bit 양자화 LLM - LunarLander-v2"
# python rl_modules/train.py agent=ppo env=lunarlander llm.mock=false llm.use_4bit=true llm.model_name="microsoft/DialoGPT-medium"

echo "=== LLM 가이드 학습 완료 ==="
echo "참고: 실제 LLM을 사용하려면 transformers와 관련 패키지를 설치하고 적절한 모델을 지정하세요."