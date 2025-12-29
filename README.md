# LLM 기반 RL 에이전트 Robustness 제어 프로젝트 (LLM-ACL)

이 프로젝트는 Large Language Model(LLM)을 활용하여 Reinforcement Learning(RL) 에이전트의 Robustness를 동적으로 제어하는 연구를 위한 코드베이스입니다. **LLM-ACL (LLM-Adaptive Curriculum Learning)** 프레임워크를 통해 에이전트의 학습 상태를 분석하고, 적절한 시점에 환경 교란(Perturbation)이나 보상 강화(Boost)를 적용하여 에이전트의 안정성과 적응성을 향상시키는 것을 목표로 합니다.

## 보고서 다운로드

본 프로젝트의 상세한 연구 내용과 실험 결과는 아래 논문/보고서 파일에서 확인하실 수 있습니다.

- [LLM-ACL: Large Language Model-Guided Adversarial Curriculum Learning for Robust Reinforcement Learning (PDF)](LLM-ACL: Large Language Model-Guided Adversarial Curriculum Learning for Robust Reinforcement Learning.pdf)

## 주요 기능

- **다양한 RL 알고리즘 지원**: SAC, PPO, TD3, A2C 알고리즘 구현
- **LLM 통합**: HuggingFace Transformers 및 vLLM 기반 LLM 로드 및 추론
- **동적 커리큘럼 제어 (LLM-ACL)**:
  - **BOOST 모드**: 학습 정체 시 보상 스케일링 및 탐험 보너스 제공
  - **MAINTAIN 모드**: 안정적인 학습을 위해 기본 환경 유지
  - **PERTURB 모드**: 강건성 향상을 위해 관측/행동 노이즈, 지연(Delay), 동역학 변화 등 다양한 교란 적용
- **양자화 지원**: 4bit/8bit 양자화로 GPU 메모리 효율성 향상
- **모의 모드**: LLM 없이 테스트 가능한 모의 모드 제공

## 설치 방법

### 1. 환경 설정
```bash
# Python 3.8+ 권장
conda create -n llm-rl python=3.10
conda activate llm-rl
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 주요 패키지
- `torch`: 딥러닝 프레임워크
- `stable-baselines3`: RL 알고리즘 라이브러리
- `transformers`: LLM 로드 및 추론
- `accelerate`, `bitsandbytes`: LLM 최적화
- `gymnasium`: 강화학습 환경
- `hydra-core`: 설정 관리

## 프로젝트 구조

```
DL_Project/
├── configs/                 # Hydra 설정 파일들
│   ├── config.yaml         # 메인 설정
│   ├── agent/              # 에이전트별 하이퍼파라미터
│   ├── env/                # 환경 설정
│   └── llm/                # LLM 설정
├── llm_core/               # LLM 관련 모듈
│   ├── loader.py           # 모델 로드
│   ├── decider.py          # 의사결정 로직
│   └── __init__.py
├── rl_modules/             # RL 학습 모듈
│   ├── train.py            # 메인 학습 스크립트
│   ├── utils.py            # 환경 유틸리티
│   ├── custom_wrappers.py  # Robustness 래퍼
│   └── custom_callbacks.py # LLM 콜백
├── scripts/                # 실행 스크립트
│   ├── run_baseline.sh     # 베이스라인 학습
│   └── run_llm_robust.sh   # LLM 적용 학습
├── outputs/                # 학습 결과 저장
└── requirements.txt        # 의존성 목록
```

## 실행 방법

**참고**: 모든 학습 실행 시 자동으로 Weights & Biases (wandb)에 실험 결과가 기록됩니다.
프로젝트: `(본인 설정 필요)`
태그: 알고리즘, 환경, LLM 사용 여부

### 1. 베이스라인 실험 (LLM 미적용)
기본 RL 알고리즘의 성능을 측정하기 위한 베이스라인 실험을 실행합니다.

```bash
# 전체 베이스라인 실행 (SAC, PPO, TD3, A2C × MuJoCo, LunarLander)
bash scripts/run_baseline.sh

# 개별 알고리즘 실행 예시 (LLM 완전 비활성화)
python rl_modules/train.py agent=sac env=mujoco llm_enabled=false
```

### 2. LLM Robustness 실험
LLM을 활용한 동적 Robustness 제어를 적용한 실험을 실행합니다.

```bash
# 모의 LLM 모드로 테스트
bash scripts/run_llm_robust.sh

# 실제 LLM 사용 (GPT-2 예시)
python rl_modules/train.py agent=sac env=mujoco llm.mock=false llm.model_name="gpt2"

# 4bit 양자화 적용
python rl_modules/train.py agent=ppo env=lunarlander llm.use_4bit=true llm.model_name="microsoft/DialoGPT-medium"
```

### 3. 모델 평가 및 비디오 녹화
학습된 모델의 성능을 평가하고 비디오로 녹화할 수 있습니다.

```bash
# 기본 평가 (점수만 출력)
python rl_modules/evaluate.py --model_path outputs/HalfCheetah-v4/sac/20251205_120000/final_model.zip --env_name HalfCheetah-v4

# 비디오 녹화 포함 평가
python rl_modules/evaluate.py --model_path outputs/HalfCheetah-v4/sac/20251205_120000/final_model.zip --env_name HalfCheetah-v4 --render_video --episodes 3

# 스크립트 사용
bash scripts/evaluate_model.sh --model_path outputs/HalfCheetah-v4/sac/20251205_120000/final_model.zip --env_name HalfCheetah-v4 --render_video
```

### 4. 사전 학습된 모델 평가 (Reproducibility)

사전 학습된 모델의 `outputs` 폴더를 다운로드하여 전체 모델을 평가할 수 있습니다.

1. [Google Drive 링크](https://drive.google.com/drive/folders/1oN9elESMlFl-4YLPkM7gSjdrVAxKMnQ2?usp=sharing)에서 `outputs` 폴더를 다운로드하여 프로젝트 루트 경로에 위치시킵니다.
2. 아래 스크립트를 실행하여 모든 모델에 대한 평가를 수행합니다.

```bash
python scripts/evaluate_all_models.py
```

## GPU 사용 설정

### 자동 GPU 할당 (권장)
프로젝트는 기본적으로 GPU를 자동으로 감지하여 사용합니다:

```bash
# 기본 설정 (자동 GPU 할당)
python rl_modules/train.py

# 특정 GPU 지정
python rl_modules/train.py device="cuda:0"

# CPU 강제 사용
python rl_modules/train.py device="cpu"
```

### 환경 변수로 GPU 지정
특정 GPU를 명시적으로 할당하려면 `CUDA_VISIBLE_DEVICES`를 사용하세요:

```bash
# GPU 0만 사용
CUDA_VISIBLE_DEVICES=0 python rl_modules/train.py

# GPU 1만 사용
CUDA_VISIBLE_DEVICES=1 python rl_modules/train.py

# 여러 GPU 사용 (쉼표로 구분)
CUDA_VISIBLE_DEVICES=0,1 python rl_modules/train.py
```

### 스크립트에서 GPU 설정
실행 스크립트에서도 GPU를 지정할 수 있습니다 (기본적으로 GPU 0 사용):

```bash
# 베이스라인 스크립트 실행 (GPU 0 자동 사용)
bash scripts/run_baseline.sh

# 다른 GPU 사용 시 스크립트 내 CUDA_VISIBLE_DEVICES 값 변경
# 또는 명령줄에서 override
CUDA_VISIBLE_DEVICES=1 bash scripts/run_baseline.sh

# LLM 실험 실행
bash scripts/run_llm_robust.sh
```

### GPU 메모리 확인
학습 중 GPU 메모리 사용량을 모니터링하려면:

```bash
# 별도 터미널에서 실행
watch -n 1 nvidia-smi
```

### 4. 설정 커스터마이징
Hydra를 사용하여 다양한 설정을 쉽게 변경할 수 있습니다.

```bash
# 다른 알고리즘과 환경 조합
python rl_modules/train.py agent=ppo env=lunarlander

# 학습 및 체크포인트 주기 변경
python rl_modules/train.py total_timesteps=500000 checkpoint_freq=5000

# LLM 설정 변경
python rl_modules/train.py llm.temperature=0.8 llm.max_length=256
```

## 설정 파일 설명

### 에이전트 설정 (`configs/agent/`)
- `sac.yaml`: SAC 알고리즘 하이퍼파라미터
- `ppo.yaml`: PPO 알고리즘 하이퍼파라미터
- `td3.yaml`: TD3 알고리즘 하이퍼파라미터
- `a2c.yaml`: A2C 알고리즘 하이퍼파라미터

### 환경 설정 (`configs/env/`)
- `mujoco.yaml`: MuJoCo 환경 (HalfCheetah-v4)
- `lunarlander.yaml`: LunarLander 환경

### LLM 설정 (`configs/llm/`)
- `default.yaml`: 기본 LLM 설정 (모델, 양자화, 모의 모드 등)

## 연구 방법론 (LLM-ACL 프레임워크)

1. **에이전트 상태 분석**: LLM이 최근 에피소드의 보상, 길이, 성공 여부 등을 분석
2. **모드 결정**:
   - **BOOST**: 성능이 정체되거나 초기 학습 단계일 때, 보상 스케일링이나 탐험 보너스를 통해 학습 촉진
   - **MAINTAIN**: 학습이 안정적으로 진행 중일 때, 현재 환경 설정을 유지
   - **PERTURB**: 에이전트가 충분히 숙련되었을 때, 다양한 노이즈(Action/Observation Noise, Delay, Dynamics Shift)를 주입하여 강건성 훈련
3. **플랜 생성 및 적용**: 결정된 모드에 따라 구체적인 파라미터(노이즈 크기, 지연 시간 등)를 포함한 플랜을 생성하고 환경 래퍼(`ALRTEnvWrapper`)에 적용
4. **반복적 개선**: 학습 과정 동안 주기적으로 이 과정을 반복하여 점진적으로 더 강건한 에이전트 육성
5. **실험 추적**: Weights & Biases를 통한 학습 메트릭 및 모델 버전 관리

## 모델 저장 및 결과 관리

### 저장 구조
학습된 모델과 체크포인트는 다음과 같은 폴더 구조로 저장됩니다:
```
outputs/
├── {환경이름}/
│   ├── {알고리즘이름}/
│   │   ├── {YYYYMMDD_HHMMSS}/  # 타임스탬프별 실험 폴더
│   │   │   ├── checkpoint_10000_steps.zip  # 주기적 체크포인트
│   │   │   ├── checkpoint_20000_steps.zip
│   │   │   ├── final_model.zip              # 최종 모델
│   │   │   └── videos/                      # 평가 비디오 저장 폴더
│   │   │       ├── sac_HalfCheetah-v4-episode-0.mp4
│   │   │       └── ...
```

### 저장 주기
- **체크포인트**: `checkpoint_freq` 설정(기본 10,000 스텝)마다 자동 저장
- **최종 모델**: 학습 완료 후 저장
- **Hydra 설정**: 각 실험 폴더에 `.hydra/` 서브폴더에 설정 파일 백업

## 주의사항

- 실제 LLM 사용 시 GPU 메모리가 충분한지 확인하세요
- 4bit/8bit 양자화는 메모리 사용을 줄여주지만 성능에 영향을 줄 수 있습니다
- 모의 모드(`llm.mock=true`)로 먼저 테스트하는 것을 권장합니다
- 학습 결과는 `outputs/` 폴더에 환경/알고리즘/날짜별로 체계적으로 저장됩니다
- 베이스라인 실험 시 `llm_enabled=false`로 설정하여 LLM 관련 오버헤드를 제거하세요

## 기여 방법

1. 이슈 생성 또는 풀 리퀘스트 제출
2. 새로운 RL 알고리즘 또는 환경 추가
3. LLM 모델 다양화
4. 실험 결과 분석 및 시각화 기능 추가

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.