import numpy as np
import gymnasium as gym
from gymnasium import ActionWrapper

class LLMGuidedRobustnessWrapper(ActionWrapper):
    """
    LLM이 정한 스케줄에 따라 동적으로 다양한 공격을 수행하는 래퍼
    """
    def __init__(self, env, action_dim: int):
        super().__init__(env)
        self.action_dim = action_dim
        self.current_step = 0
        
        # 공격 스케줄 저장소
        # 형식: [{'start': 0, 'end': 100, 'type': 'noise', 'target': [0, 1], 'params': 0.5}, ...]
        self.attack_schedule = [] 

    def update_schedule(self, schedule: list):
        """LLM으로부터 새로운 공격 스케줄을 받습니다."""
        self.attack_schedule = schedule
        print(f"[Wrapper] 새로운 공격 스케줄 적용됨 (총 {len(schedule)}개 구간)")

    def reset(self, **kwargs):
        """에피소드 시작 시 스텝 초기화"""
        self.current_step = 0
        return super().reset(**kwargs)

    def action(self, action):
        """현재 스텝에 맞는 공격을 찾아서 수행"""
        modified_action = action.copy()
        
        # 현재 스텝에 해당하는 공격 찾기
        for plan in self.attack_schedule:
            if plan['start'] <= self.current_step < plan['end']:
                modified_action = self._apply_attack(modified_action, plan)
        
        self.current_step += 1
        return modified_action

    def _apply_attack(self, action, plan):
        """공격 유형별 로직 적용"""
        attack_type = plan.get('type', 'none')
        targets = plan.get('target', []) # 공격할 액션 인덱스 리스트
        val = plan.get('value', 0.0)     # 공격 강도 파라미터

        # 타겟 인덱스에 대해서만 반복
        for idx in targets:
            if idx >= self.action_dim: continue

            if attack_type == 'mask':
                # 해당 액션을 0으로 만듦 (기존 Masking)
                action[idx] = 0.0
            
            elif attack_type == 'noise':
                # Gaussian Noise 추가
                noise = np.random.normal(0, val)
                action[idx] += noise
            
            elif attack_type == 'scale':
                # 액션 크기 증폭/감소 (예: 힘을 2배로 하거나 절반으로)
                action[idx] *= val
            
            elif attack_type == 'fix':
                # 특정 값으로 고정 (예: 1.0으로 풀파워 고정)
                action[idx] = val

        return action