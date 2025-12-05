"""
LLM 의사결정 모듈
RL 에이전트의 상태를 입력받아 LLM을 통해 Action Dropout Mask를 결정합니다.
"""

import re
import random
import torch
from .loader import LLMLoader


class LLMDecider:
    """
    LLM을 사용하여 RL 에이전트의 Action Dropout Mask를 결정하는 클래스
    """

    def __init__(self, model_name: str, use_4bit: bool = False, use_8bit: bool = False, mock: bool = False):
        """
        LLMDecider 초기화

        Args:
            model_name (str): 사용할 LLM 모델 이름
            use_4bit (bool): 4bit 양자화 사용
            use_8bit (bool): 8bit 양자화 사용
            mock (bool): 모의 모드 사용 (LLM 없이 랜덤 응답)
        """
        self.mock = mock
        self.model_name = model_name

        if not self.mock:
            self.loader = LLMLoader(model_name, use_4bit, use_8bit)
            self.tokenizer, self.model = self.loader.load()
        else:
            self.tokenizer = None
            self.model = None

    def decide_action_dropout(self, state_text: str, action_dim: int) -> list[int]:
        """
        주어진 상태 텍스트를 기반으로 Action Dropout Mask를 결정합니다.

        Args:
            state_text (str): RL 에이전트의 현재 상태 설명
            action_dim (int): 액션 차원의 크기

        Returns:
            list[int]: Action Dropout Mask (1: 유지, 0: 드롭아웃)
        """
        if self.mock:
            # 모의 모드: 랜덤 마스크 생성
            return [random.choice([0, 1]) for _ in range(action_dim)]

        # 프롬프트 구성
        prompt = f"""
다음은 RL 에이전트의 현재 상태입니다:

{state_text}

이 상태에서 액션 차원의 Robustness를 위해 어떤 액션을 드롭아웃할지 결정해주세요.
응답 형식: [1, 0, 1, 1] (1은 유지, 0은 드롭아웃, 총 {action_dim}개 숫자)
"""

        # 입력 토큰화
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # 텍스트 생성
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # 응답 디코딩
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 마스크 파싱
        mask = self._parse_mask(response, action_dim)

        return mask

    def _parse_mask(self, response: str, action_dim: int) -> list[int]:
        """
        LLM 응답에서 Action Dropout Mask를 파싱합니다.

        Args:
            response (str): LLM의 응답 텍스트
            action_dim (int): 예상되는 마스크 길이

        Returns:
            list[int]: 파싱된 마스크 또는 기본값
        """
        # 정규식으로 [1,0,1,1] 형식의 리스트 찾기
        match = re.search(r'\[([01](?:,\s*[01])*\)]', response)

        if match:
            mask_str = match.group(0)
            try:
                # 문자열을 리스트로 변환 (주의: eval 사용은 보안상 위험할 수 있음)
                mask = eval(mask_str)
                # 유효성 검증
                if (len(mask) == action_dim and
                    all(isinstance(x, int) and x in [0, 1] for x in mask)):
                    return mask
            except (ValueError, SyntaxError):
                pass

        # 파싱 실패 시 기본값 반환 (모든 액션 유지)
        return [1] * action_dim