#!/bin/bash

# 모든 모델 평가 및 강건성 테스트 실행 스크립트
# outputs/ 폴더의 모든 모델을 순회하며 평가하고 CSV로 결과 저장

# Python 경로 설정
export PYTHONPATH="/home/mlic/mingukang/DL_Project:$PYTHONPATH"

# GPU 설정 (필요시 변경)
export CUDA_VISIBLE_DEVICES=0

echo "=== 전체 모델 평가 스크립트 ==="

# 기본 출력 파일 경로
OUTPUT_CSV="model_evaluation_results.csv"

# 사용자 지정 출력 파일 경로 (옵션)
if [ ! -z "$1" ]; then
    OUTPUT_CSV="$1"
fi

echo "결과 저장 경로: $OUTPUT_CSV"
echo ""

# Python 스크립트 실행
python scripts/evaluate_all_models.py --output "$OUTPUT_CSV" --episodes 5

echo ""
echo "평가 완료! 결과: $OUTPUT_CSV"
