import os
from vllm import LLM, SamplingParams

class LLMLoader:
    """
    vLLM을 사용하여 대규모 언어 모델(LLM)을 로드하고 추론하는 클래스.
    A100 8장 환경에 맞춰 Tensor Parallelism을 지원합니다.
    """

    def __init__(self, model_id: str, tensor_parallel_size: int = 8, gpu_memory_utilization: float = 0.90):
        """
        초기화 메서드
        
        Args:
            model_id (str): HuggingFace 모델 ID (예: meta-llama/Meta-Llama-3.1-70B-Instruct)
            tensor_parallel_size (int): 사용할 GPU 개수 (A100 8장이면 8)
            gpu_memory_utilization (float): GPU 메모리 점유율 설정 (0.9 ~ 0.95 추천)
        """
        self.model_id = model_id
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = None
        self.tokenizer = None

    def load(self):
        """
        vLLM 엔진을 로드합니다.
        """
        print(f"Loading model: {self.model_id} with TP={self.tensor_parallel_size}...")
        
        # vLLM 인스턴스 생성
        self.llm = LLM(
            model=self.model_id,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            # 만약 FP8 양자화를 쓰고 싶다면 아래 주석 해제 (H100/A100 권장)
            # quantization="fp8", 
        )
        
        # 토크나이저는 vLLM 내부에서 가져옵니다.
        self.tokenizer = self.llm.get_tokenizer()
        print("Model loaded successfully.")
        return self.llm

    def generate_response(self, conversation: list, temperature: float = 0.7, max_tokens: int = 512):
        """
        대화 리스트(List[Dict])를 받아 응답을 생성합니다.
        
        Args:
            conversation (list): OpenAI 스타일의 대화 목록 [{"role": "user", ...}]
            temperature (float): 창의성 조절
            max_tokens (int): 최대 생성 토큰 수
        
        Returns:
            str: 생성된 텍스트 응답
        """
        if self.llm is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        # 1. 채팅 템플릿 적용
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # 2. 샘플링 파라미터 설정
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens
        )

        # 3. 추론 실행
        outputs = self.llm.generate([prompt], sampling_params)

        # 4. 결과 반환
        return outputs[0].outputs[0].text

# --- 사용 예시 (Main) ---
if __name__ == "__main__":
    # 1. 설정
    # Llama 3.1 70B 모델 (A100 8장이면 FP16 원본도 충분히 돌아갑니다)
    MODEL_ID = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    
    # 2. 클래스 인스턴스 생성 (GPU 8장 사용)
    loader = LLMLoader(model_id=MODEL_ID, tensor_parallel_size=8)
    
    # 3. 모델 로드
    loader.load()

    # 4. 대화 생성
    messages = [
        {"role": "system", "content": "너는 친절하고 유능한 AI 어시스턴트야."},
        {"role": "user", "content": "A100 GPU 8장으로 할 수 있는 멋진 인공지능 프로젝트 3가지만 추천해줘."}
    ]

    print("\n" + "="*50)
    print(f"질문: {messages[-1]['content']}")
    print("-" * 50)
    
    response = loader.generate_response(messages)
    
    print(f"답변:\n{response}")
    print("="*50 + "\n")