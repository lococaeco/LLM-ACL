"""
LLM 모델 로더 모듈
HuggingFace transformers를 사용하여 LLM 모델과 토크나이저를 로드합니다.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


class LLMLoader:
    """
    LLM 모델과 토크나이저를 로드하는 클래스
    4bit 또는 8bit 양자화를 지원하여 GPU 메모리 효율을 높입니다.
    """

    def __init__(self, model_name: str, use_4bit: bool = False, use_8bit: bool = False):
        """
        LLMLoader 초기화

        Args:
            model_name (str): 로드할 모델의 HuggingFace 모델 이름
            use_4bit (bool): 4bit 양자화 사용 여부
            use_8bit (bool): 8bit 양자화 사용 여부
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        self.use_8bit = use_8bit
        self.tokenizer = None
        self.model = None

        # 양자화 옵션 검증
        if self.use_4bit and self.use_8bit:
            raise ValueError("4bit와 8bit 양자화를 동시에 사용할 수 없습니다.")

    def load(self):
        """
        모델과 토크나이저를 로드합니다.

        Returns:
            tuple: (tokenizer, model)
        """
        # 양자화 설정 구성
        quantization_config = None
        if self.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.use_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # PAD 토큰 설정 (없는 경우 EOS 토큰 사용)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 모델 로드
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto",  # 자동으로 GPU 할당
            torch_dtype=torch.float16 if quantization_config else torch.float32
        )

        return self.tokenizer, self.model