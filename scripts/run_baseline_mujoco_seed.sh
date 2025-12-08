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

# 각 환경에 대해 반복
for env in "${environments[@]}"; do
    echo "=== $env 환경 학습 시작 ==="

    # 각 시드에 대해 반복
    for seed in "${seeds[@]}"; do
        echo "시드 $seed 로 학습 시작"

        # 각 알고리즘을 병렬로 실행 (GPU 3,5,6,7)
        echo "SAC 알고리즘 - $env (GPU 3, 시드 $seed)"
        CUDA_VISIBLE_DEVICES=3 python rl_modules/train.py agent=sac env=$env llm_enabled=false seed=$seed &

        echo "PPO 알고리즘 - $env (GPU 5, 시드 $seed)"
        CUDA_VISIBLE_DEVICES=5 python rl_modules/train.py agent=ppo env=$env llm_enabled=false seed=$seed &

        echo "TD3 알고리즘 - $env (GPU 6, 시드 $seed)"
        CUDA_VISIBLE_DEVICES=6 python rl_modules/train.py agent=td3 env=$env llm_enabled=false seed=$seed &

        echo "A2C 알고리즘 - $env (GPU 7, 시드 $seed)"
        CUDA_VISIBLE_DEVICES=7 python rl_modules/train.py agent=a2c env=$env llm_enabled=false seed=$seed &

        # 현재 시드의 모든 알고리즘 학습 완료 대기
        wait

        echo "시드 $seed 학습 완료"
    done

    echo "=== $env 환경 모든 시드 학습 완료 ==="
done

echo "=== 모든 베이스라인 학습 완료 ==="