#!/bin/bash

# 모델 평가 및 비디오 녹화 스크립트
# 학습된 모델을 평가하고 선택적으로 비디오를 녹화합니다.

# Python 경로 설정
export PYTHONPATH="/home/mlic/mingukang/DL_Project:$PYTHONPATH"

# GPU 설정 (필요시 변경: 0, 1, 또는 all)
export CUDA_VISIBLE_DEVICES=0

echo "=== RL 모델 평가 스크립트 ==="

# 사용법 표시
if [ $# -eq 0 ]; then
    echo "사용법: $0 --model_path <모델경로> --env_name <환경이름> [옵션들]"
    echo ""
    echo "필수 인자:"
    echo "  --model_path    평가할 모델 파일 경로"
    echo "  --env_name      환경 이름 (HalfCheetah-v4, LunarLander-v2 등)"
    echo ""
    echo "선택 인자:"
    echo "  --episodes      평가 에피소드 수 (기본: 5)"
    echo "  --render_video  비디오 녹화 활성화"
    echo "  --video_folder  비디오 저장 폴더 (기본: 모델폴더/videos)"
    echo ""
    echo "예시:"
    echo "  $0 --model_path outputs/HalfCheetah-v4/sac/20251205_120000/final_model.zip --env_name HalfCheetah-v4 --render_video"
    exit 1
fi

# 평가 실행
python rl_modules/evaluate.py "$@"

echo "평가가 완료되었습니다."