#!/bin/bash
set -euo pipefail

# 베이스라인 RL 학습 실행 스크립트 (다중 시드)
# 각 환경당 4개의 시드로 실험을 진행합니다.

# Python 경로 설정
export PYTHONPATH="/home/mlic/mingukang/DL_Project:$PYTHONPATH"

echo "=== 베이스라인 RL 학습 시작 (다중 시드) ==="

# 시드 배열
seeds=(1 123 456 789)

# 환경 배열
environments=("mujoco" "hopper" "ant" "humanoid" "lunarlander")

# 알고리즘 배열
models=("sac" "ppo" "td3" "a2c")

# GPU 배열 (시드 순서대로 매핑)
gpus=(3 5 6 7)

# 각 환경에 대해 반복
for env in "${environments[@]}"; do
    echo "=== $env 환경 학습 시작 ==="

    # 각 알고리즘에 대해 반복
    for model in "${models[@]}"; do
        echo "=== $model 알고리즘 학습 시작 ==="

        # 4개의 시드를 병렬로 실행 (GPU 3,5,6,7)
        for i in "${!seeds[@]}"; do
            seed=${seeds[$i]}
            gpu=${gpus[$i]}
            
            echo "$model 알고리즘 - $env (GPU $gpu, 시드 $seed)"
            CUDA_VISIBLE_DEVICES=$gpu python rl_modules/train.py agent=$model env=$env llm_enabled=false seed=$seed &
        done

        # 현재 모델의 모든 시드 학습 완료 대기
        wait
        echo "$model 알고리즘 학습 완료"
    done

    echo "=== $env 환경 모든 모델 학습 완료 ==="
done

echo "=== 모든 베이스라인 학습 완료 ==="