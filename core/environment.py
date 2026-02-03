"""
环境模块 - 提供学习任务的交互界面
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import numpy as np

class Environment(ABC):
    """环境基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.type = config.get('type', 'unknown')
        self.description = config.get('description', '')
        
    @abstractmethod
    def reset(self) -> Any:
        """重置环境状态"""
        pass
    
    @abstractmethod
    def step(self, action: Any) -> Tuple[Any, Any, bool, Dict]:
        """
        执行一步环境交互
        
        Returns:
            observation: 观察状态
            feedback: 反馈信号（非梯度）
            done: 是否完成
            info: 额外信息
        """
        pass
    
    @abstractmethod
    def get_feedback(self, prediction: Any, target: Any) -> Any:
        """获取反馈信号（非梯度）"""
        pass


class SequencePredictionEnvironment(Environment):
    """序列预测环境 - Phase 1 使用"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # 序列配置
        self.sequence_length = config.get('sequence_length', 10)
        self.vocabulary_size = config.get('vocabulary_size', 5)
        self.current_step = 0
        self.sequence = None
        self.target = None
        
    def reset(self) -> np.ndarray:
        """重置序列"""
        self.current_step = 0
        
        # 生成简单结构化序列
        # 例如: [0, 1, 0, 1, 2, 3, 2, 3, 4, 4]
        pattern_length = self.sequence_length // 2
        pattern = np.random.randint(0, self.vocabulary_size, pattern_length)
        self.sequence = np.concatenate([pattern, pattern])
        
        # 当前观察（历史窗口）
        window_size = min(3, self.sequence_length - 1)
        observation = self.sequence[self.current_step:self.current_step + window_size]
        
        # 目标（下一个符号）
        if self.current_step + window_size < len(self.sequence):
            self.target = self.sequence[self.current_step + window_size]
        else:
            self.target = None
        
        return observation
    
    def step(self, prediction: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步预测
        
        Args:
            prediction: 预测的下一个符号
            
        Returns:
            observation: 新的观察窗口
            feedback: 正确性反馈（0或1）
            done: 是否序列结束
            info: 额外信息
        """
        if self.target is None:
            raise ValueError("环境未重置或序列已结束")
        
        # 计算反馈（正确性）
        correct = 1.0 if prediction == self.target else 0.0
        
        # 更新步骤
        self.current_step += 1
        
        # 检查是否完成
        done = self.current_step >= self.sequence_length - 1
        
        # 获取新的观察
        window_size = min(3, self.sequence_length - 1 - self.current_step)
        if window_size > 0:
            observation = self.sequence[self.current_step:self.current_step + window_size]
        else:
            observation = np.array([])
        
        # 设置新的目标
        if self.current_step + window_size < len(self.sequence):
            self.target = self.sequence[self.current_step + window_size]
        else:
            self.target = None
        
        info = {
            "step": self.current_step,
            "target": self.target,
            "prediction": prediction,
            "correct": bool(correct)
        }
        
        return observation, correct, done, info
    
    def get_feedback(self, prediction: Any, target: Any) -> float:
        """获取简单正确性反馈"""
        return 1.0 if prediction == target else 0.0