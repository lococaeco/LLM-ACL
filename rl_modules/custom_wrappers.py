"""
커스텀 환경 래퍼 모듈
LLM의 지도를 받아 RL 에이전트의 Robustness를 제어하는 래퍼들을 제공합니다.
"""

import numpy as np
from gymnasium import ActionWrapper


class LLMGuidedRobustnessWrapper(ActionWrapper):
    """
    LLM의 지도를 받아 액션에 Dropout Mask와 노이즈를 적용하는 환경 래퍼

    이 래퍼는 외부에서 dropout_mask와 noise_scale을 업데이트할 수 있으며,
    이를 통해 RL 에이전트의 행동을 동적으로 변형하여 Robustness를 제어합니다.
    """

    def __init__(self, env, action_dim: int):
        """
        LLMGuidedRobustnessWrapper 초기화

        Args:
            env: 래핑할 환경 객체
            action_dim (int): 액션 차원의 크기
        """
        super().__init__(env)
        self.action_dim = action_dim
        self.dropout_mask = [1] * action_dim  # 기본값: 모든 액션 유지
        self.noise_scale = 0.0  # 기본값: 노이즈 없음

    def update_mask(self, mask: list[int]):
        """
        Dropout Mask를 업데이트합니다.

        Args:
            mask (list[int]): 새로운 Dropout Mask (1: 유지, 0: 드롭아웃)
        """
        if len(mask) == self.action_dim:
            self.dropout_mask = mask
        else:
            print(f"경고: 마스크 길이({len(mask)})가 액션 차원({self.action_dim})과 일치하지 않습니다.")

    def update_noise_scale(self, scale: float):
        """
        노이즈 스케일을 업데이트합니다.

        Args:
            scale (float): 새로운 노이즈 스케일
        """
        self.noise_scale = scale

    def action(self, action):
        """
        액션에 Dropout Mask와 노이즈를 적용합니다.

        Args:
            action: 원본 액션

        Returns:
            변형된 액션
        """
        # 액션을 numpy 배열로 변환
        action_array = np.array(action, dtype=np.float32)

        # Dropout Mask 적용
        masked_action = action_array * np.array(self.dropout_mask, dtype=np.float32)

        # 노이즈 추가 (옵션)
        if self.noise_scale > 0:
            noise = np.random.normal(0, self.noise_scale, size=masked_action.shape)
            masked_action += noise

        return masked_action