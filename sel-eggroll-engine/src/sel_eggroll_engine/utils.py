# -*- coding: utf-8 -*-
"""
工具函数模块

提供数据生成、性能评估等辅助功能。
"""

import numpy as np
from typing import Dict, Tuple, List


def generate_sensor_data(n_samples: int = 1000, anomaly_ratio: float = 0.1, seed: int = None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """生成传感器数据
    
    Args:
        n_samples: 样本数量
        anomaly_ratio: 异常比例
        seed: 随机种子
        
    Returns:
        传感器数据字典，格式为 {sensor_type: (X, y)}
    """
    rng = np.random.default_rng(seed)
    data = {}
    
    # 温度数据
    temp_data = []
    temp_labels = []
    base_temp = 22.0
    for i in range(n_samples):
        if rng.random() > anomaly_ratio:
            temp = base_temp + rng.normal(0, 1.0)
            temp_data.append([temp, i % 24, base_temp])
            temp_labels.append([1, 0])  # 正常
            base_temp = temp
        else:
            temp = base_temp + rng.normal(5, 2.0)
            temp_data.append([temp, i % 24, base_temp])
            temp_labels.append([0, 1])  # 异常
            base_temp = temp
    data['temperature'] = (np.array(temp_data), np.array(temp_labels))
    
    # 湿度数据
    hum_data = []
    hum_labels = []
    base_hum = 45.0
    for i in range(n_samples):
        if rng.random() > anomaly_ratio:
            hum = base_hum + rng.normal(0, 3.0)
            hum_data.append([hum, i % 24, base_hum])
            hum_labels.append([1, 0])
            base_hum = hum
        else:
            hum = base_hum + rng.normal(20, 5.0)
            hum_data.append([hum, i % 24, base_hum])
            hum_labels.append([0, 1])
            base_hum = hum
    data['humidity'] = (np.array(hum_data), np.array(hum_labels))
    
    # 运动数据
    motion_data = []
    motion_labels = []
    base_motion = 0.0
    for i in range(n_samples):
        if rng.random() > anomaly_ratio:
            motion = base_motion + rng.normal(0, 0.1)
            motion_data.append([motion, i % 24, base_motion])
            motion_labels.append([1, 0])
            base_motion = motion
        else:
            motion = base_motion + rng.normal(1.0, 0.3)
            motion_data.append([motion, i % 24, base_motion])
            motion_labels.append([0, 1])
            base_motion = motion
    data['motion'] = (np.array(motion_data), np.array(motion_labels))
    
    # 门数据
    door_data = []
    door_labels = []
    base_door = 0.0
    for i in range(n_samples):
        if rng.random() > anomaly_ratio:
            door = base_door
            door_data.append([door, i % 24, base_door])
            door_labels.append([1, 0])
            if rng.random() < 0.05:  # 正常开关
                base_door = 1.0 - base_door
        else:
            door = 1.0 - base_door
            door_data.append([door, i % 24, base_door])
            door_labels.append([0, 1])
            base_door = door
    data['door'] = (np.array(door_data), np.array(door_labels))
    
    # 多模态标签（综合异常）
    multimodal_labels = []
    for i in range(n_samples):
        is_anomaly = False
        for sensor_type in ['temperature', 'humidity', 'motion', 'door']:
            _, y = data[sensor_type]
            if np.argmax(y[i]) == 1:
                is_anomaly = True
                break
        multimodal_labels.append([0, 1] if is_anomaly else [1, 0])
    data['multimodal'] = (np.array(multimodal_labels), np.array(multimodal_labels))
    
    return data


def generate_xor_data(n_samples: int = 1000, noise: float = 0.1, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """生成 XOR 数据
    
    Args:
        n_samples: 样本数量
        noise: 噪声水平
        seed: 随机种子
        
    Returns:
        (X, y)：输入数据和标签
    """
    rng = np.random.default_rng(seed)
    X = rng.rand(n_samples, 2) * 2 - 1
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(int)
    y = np.eye(2)[y]
    X += rng.normal(0, noise, X.shape)
    return X, y


def generate_spiral_data(n_samples: int = 1000, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """生成螺旋数据
    
    Args:
        n_samples: 样本数量
        seed: 随机种子
        
    Returns:
        (X, y)：输入数据和标签
    """
    rng = np.random.default_rng(seed)
    n = n_samples // 2
    theta = np.linspace(0, 4 * np.pi, n)
    r = np.linspace(0.0, 1.0, n)
    
    X1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    X2 = np.column_stack([r * np.cos(theta + np.pi), r * np.sin(theta + np.pi)])
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(n), np.ones(n)]).astype(int)
    y = np.eye(2)[y]
    
    # 添加噪声
    X += rng.normal(0, 0.05, X.shape)
    
    # 打乱数据
    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def generate_regression_data(n_samples: int = 1000, noise: float = 0.1, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """生成回归数据
    
    Args:
        n_samples: 样本数量
        noise: 噪声水平
        seed: 随机种子
        
    Returns:
        (X, y)：输入数据和标签
    """
    rng = np.random.default_rng(seed)
    X = rng.rand(n_samples, 1) * 10
    y = 2 * X + 3 + rng.normal(0, noise, X.shape)
    return X, y


def evaluate_performance(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """评估模型性能
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        性能指标字典
    """
    # 分类任务
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    
    # 准确率
    accuracy = np.mean(y_true == y_pred)
    
    # 精确率和召回率
    unique_classes = np.unique(y_true)
    precision = {}
    recall = {}
    
    for cls in unique_classes:
        true_positive = np.sum((y_true == cls) & (y_pred == cls))
        false_positive = np.sum((y_true != cls) & (y_pred == cls))
        false_negative = np.sum((y_true == cls) & (y_pred != cls))
        
        precision[cls] = true_positive / (true_positive + false_positive + 1e-8)
        recall[cls] = true_positive / (true_positive + false_negative + 1e-8)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


def normalize_data(X: np.ndarray) -> np.ndarray:
    """数据归一化
    
    Args:
        X: 输入数据
        
    Returns:
        归一化后的数据
    """
    mean = np.mean(X, axis=0, keepdims=True)
    std = np.std(X, axis=0, keepdims=True) + 1e-8
    return (X - mean) / std


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """训练测试集分割
    
    Args:
        X: 输入数据
        y: 标签
        test_size: 测试集比例
        seed: 随机种子
        
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def batch_generator(X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """批量数据生成器
    
    Args:
        X: 输入数据
        y: 标签
        batch_size: 批次大小
        
    Yields:
        批次数据和标签
    """
    n_samples = len(X)
    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        yield X[i:end], y[i:end]


def save_model(model, filepath: str):
    """保存模型
    
    Args:
        model: 模型实例
        filepath: 文件路径
    """
    import pickle
    import threading
    
    # 保存模型的必要属性，避免序列化锁对象
    model_data = {
        'config': model.config,
        'global_epoch': getattr(model, 'global_epoch', 0),
        'swarm_performance': getattr(model, 'swarm_performance', []),
        'nodes': []
    }
    
    # 保存每个节点的数据
    for node in model.nodes:
        node_data = {
            'node_id': node.node_id,
            'sensor_type': node.sensor_type,
            'in_size': node.in_size,
            'out_size': node.out_size,
            'sensor_params': node.sensor_params,
            'local_losses': node.local_losses,
            'accuracy_history': node.accuracy_history,
            'performance_score': node.performance_score,
            'energy_consumption': node.energy_consumption,
            'anomaly_threshold': node.anomaly_threshold,
            'sel_weights': node.sel_module.get_weights(),
            'eggroll_weights': node.eggroll_optimizer.get_weights()
        }
        model_data['nodes'].append(node_data)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)

def load_model(filepath: str):
    """加载模型
    
    Args:
        filepath: 文件路径
        
    Returns:
        模型实例
    """
    import pickle
    import threading
    from .optimized_edge import OptimizedEdgeSwarm, OptimizedEdgeNode
    
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    
    # 创建新的优化边缘智能群
    swarm = OptimizedEdgeSwarm(model_data['config'])
    swarm.global_epoch = model_data['global_epoch']
    swarm.swarm_performance = model_data['swarm_performance']
    
    # 重新创建节点
    for node_data in model_data['nodes']:
        # 创建节点
        node = OptimizedEdgeNode(
            node_id=node_data['node_id'],
            sensor_type=node_data['sensor_type'],
            in_size=node_data['in_size'],
            out_size=node_data['out_size'],
            config=model_data['config']
        )
        
        # 恢复节点属性
        node.local_losses = node_data['local_losses']
        node.accuracy_history = node_data['accuracy_history']
        node.performance_score = node_data['performance_score']
        node.energy_consumption = node_data['energy_consumption']
        node.anomaly_threshold = node_data['anomaly_threshold']
        
        # 恢复权重
        node.sel_module.set_weights(node_data['sel_weights'])
        node.eggroll_optimizer.set_weights(node_data['eggroll_weights'])
        
        # 添加到群中
        swarm.nodes.append(node)
    
    return swarm
