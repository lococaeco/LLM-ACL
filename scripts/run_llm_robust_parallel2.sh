#!/bin/bash
set -euo pipefail

# 베이스라인 RL 학습 실행 스크립트 (병렬 GPU 할당)
# 각 실험당 GPU 2개씩 할당하여 최대 4개 실험 동시 실행

# Python 경로 설정
export PYTHONPATH="/home/mlic/mingukang/DL_Project:${PYTHONPATH:-}"

echo "=== 베이스라인 RL 학습 시작 (시드 병렬 처리) ==="

# 실험 설정
algorithms=("a2c")
seeds=(1 123 456 789)

# 작업 실행 함수
run_job() {
    local algo=$1
    local env=$2
    local seed=$3
    local gpu_assignment=$4

    echo "[$(date '+%H:%M:%S')] 시작: ${algo^^} ${env} seed=${seed} [GPU:${gpu_assignment}]"

    # GPU 설정
    export CUDA_VISIBLE_DEVICES="$gpu_assignment"

    # 로그 파일
    local log_file="logs/${algo}_${env}_seed${seed}.log"
    mkdir -p logs

    # 실행
    if python rl_modules/train.py agent=$algo env=$env seed=$seed llm.mock=false llm.model_name="meta-llama/Meta-Llama-3-8B-Instruct" > "$log_file" 2>&1; then
        echo "[$(date '+%H:%M:%S')] 완료: ${algo^^} ${env} seed=${seed}"
    else
        echo "[$(date '+%H:%M:%S')] 실패: ${algo^^} ${env} seed=${seed} (로그 확인: $log_file)"
    fi
}

# 메인 루프
for algo in "${algorithms[@]}"; do
    echo "=================================================="
    echo "=== 알고리즘 시작: $algo ==="
    echo "=================================================="
    
    pids=()

    # 1. Humanoid (GPU 0-3)
    env="hopper"
    echo "--- 환경 배치: $env (GPU 0-3) ---"
    for i in "${!seeds[@]}"; do
        seed=${seeds[$i]}
        gpu=$((4 + i))  # 0, 1, 2, 3
        
        run_job "$algo" "$env" "$seed" "$gpu" &
        pid=$!
        pids+=($pid)
        
        echo "  -> [$env] 시드 $seed 시작 (PID: $pid, GPU: $gpu)"
        echo "  -> VLLM 초기화 안정화를 위해 10초 대기..."
        sleep 10
    done

    echo "--- 모든 작업($algo) 실행 중. 완료 대기... ---"
    
    # 현재 알고리즘의 모든 작업이 끝날 때까지 대기
    for pid in "${pids[@]}"; do
        wait $pid
    done
    
    echo "--- 알고리즘 완료: $algo ---"
    echo ""
done

echo "=== 모든 실험 완료 ==="
echo "로그 파일은 logs/ 디렉토리에서 확인하세요."