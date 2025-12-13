#!/bin/bash
set -euo pipefail

# 베이스라인 RL 학습 실행 스크립트 (다중 시드)
# 각 환경당 4개의 시드로 실험을 진행합니다.

# Python 경로 설정
export PYTHONPATH="/home/mlic/sangwoo/DL_Project:${PYTHONPATH:-}"

echo "=== 베이스라인 RL 학습 시작 (다중 시드) ==="

# 시드 배열
seeds=(1 123 456 789)

# 환경 배열
environments=("humanoid" "lunarlander")

# 알고리즘 배열
algorithms=("sac" "ppo" "td3" "a2c")

# 실행 간 딜레이 (초)
DELAY=60

# 각 환경에 대해 반복
for env in "${environments[@]}"; do
    echo "=== $env 환경 학습 시작 ==="

    # 각 시드에 대해 반복
    for seed in "${seeds[@]}"; do
        echo "시드 $seed 로 학습 시작"

        # 각 알고리즘에 대해 순차 실행
        for algo in "${algorithms[@]}"; do
            echo "${algo^^} 알고리즘 - $env (시드 $seed)"
            CUDA_VISIBLE_DEVICES=3 python rl_modules/train.py agent=$algo env=$env seed=$seed llm.mock=false llm.model_name="meta-llama/Meta-Llama-3-8B-Instruct"
            
            echo "다음 실행 전 ${DELAY}초 대기..."
            sleep $DELAY
        done

        echo "시드 $seed 학습 완료"
    done

    echo "=== $env 환경 모든 시드 학습 완료 ==="
done

echo "=== 모든 베이스라인 학습 완료 ==="

