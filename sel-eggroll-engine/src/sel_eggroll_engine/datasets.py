# -*- coding: utf-8 -*-
"""
真实数据集加载模块

支持加载真实工业和智能家居数据集：
- SKAB (Skoltech Anomaly Benchmark): 工业异常检测数据集
- UCI Occupancy Detection: 智能家居占用检测数据集
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import os
import requests
import zipfile
from io import BytesIO


@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str = "skab"  # "skab" or "occupancy"
    test_size: float = 0.2
    normalize: bool = True
    anomaly_ratio: float = 0.1
    random_seed: int = 42


class SKABDataset:
    """SKAB 工业异常检测数据集加载器
    
    SKAB (Skoltech Anomaly Benchmark) 是一个工业异常检测数据集，
    包含来自工业设备的传感器数据和对应的异常标签。
    """
    
    DATASET_URL = "https://github.com/waico/SKAB/archive/refs/heads/master.zip"
    DOWNLOAD_PATH = "data/skab"
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig(name="skab")
        self.rng = np.random.default_rng(self.config.random_seed)
        self.data: Optional[Dict[str, np.ndarray]] = None
        self.labels: Optional[Dict[str, np.ndarray]] = None
        
    def download(self) -> bool:
        """下载 SKAB 数据集"""
        print("正在下载 SKAB 数据集...")
        
        try:
            if not os.path.exists(self.DOWNLOAD_PATH):
                os.makedirs(self.DOWNLOAD_PATH, exist_ok=True)
                
                # 下载数据集
                response = requests.get(self.DATASET_URL)
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    z.extractall(self.DOWNLOAD_PATH)
                
                print("✓ SKAB 数据集下载完成")
            else:
                print("✓ SKAB 数据集已存在")
            
            return True
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            return False
    
    def load(self, use_synthetic: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """加载数据集
        
        Args:
            use_synthetic: 如果真实数据集不可用，使用合成数据
            
        Returns:
            (data, labels): 传感器数据和标签
        """
        if not self.download() and not use_synthetic:
            raise RuntimeError("无法加载真实数据集，且 use_synthetic=False")
        
        if not os.path.exists(os.path.join(self.DOWNLOAD_PATH, "SKAB-master")):
            print("使用合成数据替代真实 SKAB 数据集...")
            return self._generate_synthetic_data()
        
        return self._load_real_data()
    
    def _load_real_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """加载真实 SKAB 数据"""
        data_path = os.path.join(self.DOWNLOAD_PATH, "SKAB-master", "data")
        
        if not os.path.exists(data_path):
            return self._generate_synthetic_data()
        
        # 读取数据文件
        sensor_data = {}
        sensor_labels = {}
        
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
        
        if not csv_files:
            return self._generate_synthetic_data()
        
        # 读取第一个文件作为示例
        csv_file = csv_files[0]
        df = pd.read_csv(os.path.join(data_path, csv_file))
        
        # 提取传感器列（假设前几列是传感器数据）
        sensor_columns = df.columns[:-1]  # 假设最后一列是标签
        label_column = df.columns[-1]
        
        # 提取数据
        X = df[sensor_columns].values
        y = df[label_column].values
        
        # 转换为 one-hot 编码
        y_onehot = np.zeros((len(y), 2))
        y_onehot[np.arange(len(y)), y.astype(int)] = 1
        
        # 模拟多个传感器（从数据中提取不同特征）
        n_features = X.shape[1]
        
        # 划分到不同的传感器类型
        sensor_types = ['temperature', 'pressure', 'vibration', 'current']
        
        for i, sensor_type in enumerate(sensor_types):
            # 每个传感器使用不同的特征组合
            feature_indices = np.arange(i, n_features, len(sensor_types))
            if len(feature_indices) < 3:
                # 如果特征不够，填充
                feature_indices = np.pad(feature_indices, (0, 3 - len(feature_indices)), 
                                        mode='edge')
            
            sensor_data[sensor_type] = X[:, feature_indices[:3]]
            sensor_labels[sensor_type] = y_onehot
        
        self.data = sensor_data
        self.labels = sensor_labels
        
        return sensor_data, sensor_labels
    
    def _generate_synthetic_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """生成合成的工业传感器数据"""
        n_samples = 1000
        
        sensor_data = {}
        sensor_labels = {}
        
        # 生成四种传感器数据
        sensor_types = ['temperature', 'pressure', 'vibration', 'current']
        
        for sensor_type in sensor_types:
            # 正常数据
            X_normal = self.rng.normal(loc=25, scale=5, size=(int(n_samples * 0.9), 3))
            
            # 异常数据
            X_anomaly = self.rng.normal(loc=50, scale=10, size=(int(n_samples * 0.1), 3))
            
            # 合并
            X = np.vstack([X_normal, X_anomaly])
            
            # 标签
            y = np.zeros((n_samples, 2))
            y[:int(n_samples * 0.9), 0] = 1  # 正常
            y[int(n_samples * 0.9):, 1] = 1  # 异常
            
            # 打乱
            indices = self.rng.permutation(n_samples)
            X = X[indices]
            y = y[indices]
            
            # 归一化
            if self.config.normalize:
                X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            
            sensor_data[sensor_type] = X
            sensor_labels[sensor_type] = y
        
        self.data = sensor_data
        self.labels = sensor_labels
        
        return sensor_data, sensor_labels


class OccupancyDataset:
    """UCI 智能家居占用检测数据集加载器
    
    UCI Occupancy Detection 数据集包含来自智能家居的传感器数据，
    用于检测房间是否被占用。
    """
    
    DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00357/occupancy_data.zip"
    DOWNLOAD_PATH = "data/occupancy"
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig(name="occupancy")
        self.rng = np.random.default_rng(self.config.random_seed)
        self.data: Optional[Dict[str, np.ndarray]] = None
        self.labels: Optional[Dict[str, np.ndarray]] = None
    
    def download(self) -> bool:
        """下载 Occupancy 数据集"""
        print("正在下载 UCI Occupancy 数据集...")
        
        try:
            if not os.path.exists(self.DOWNLOAD_PATH):
                os.makedirs(self.DOWNLOAD_PATH, exist_ok=True)
                
                response = requests.get(self.DATASET_URL)
                with zipfile.ZipFile(BytesIO(response.content)) as z:
                    z.extractall(self.DOWNLOAD_PATH)
                
                print("✓ UCI Occupancy 数据集下载完成")
            else:
                print("✓ UCI Occupancy 数据集已存在")
            
            return True
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            return False
    
    def load(self, use_synthetic: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """加载数据集"""
        if not self.download() and not use_synthetic:
            raise RuntimeError("无法加载真实数据集，且 use_synthetic=False")
        
        data_file = os.path.join(self.DOWNLOAD_PATH, "datatraining.txt")
        
        if not os.path.exists(data_file):
            print("使用合成数据替代真实 Occupancy 数据集...")
            return self._generate_synthetic_data()
        
        return self._load_real_data()
    
    def _load_real_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """加载真实 Occupancy 数据"""
        data_file = os.path.join(self.DOWNLOAD_PATH, "datatraining.txt")
        df = pd.read_csv(data_file)
        
        # 提取传感器列
        # 列: date, Temperature, Humidity, Light, CO2, HumidityRatio, Occupancy
        sensor_columns = ['Temperature', 'Humidity', 'Light', 'CO2']
        label_column = 'Occupancy'
        
        X = df[sensor_columns].values
        y = df[label_column].values
        
        # 转换为 one-hot 编码
        y_onehot = np.zeros((len(y), 2))
        y_onehot[np.arange(len(y)), y.astype(int)] = 1
        
        # 分配到不同传感器
        sensor_data = {}
        sensor_labels = {}
        
        # 温度和湿度
        sensor_data['temperature'] = X[:, [0, 1, 4]]  # Temperature, Humidity, HumidityRatio
        sensor_labels['temperature'] = y_onehot
        
        # 光照和 CO2
        sensor_data['light'] = X[:, [2, 3, 0]]  # Light, CO2, Temperature
        sensor_labels['light'] = y_onehot
        
        # 运动（模拟）
        sensor_data['motion'] = X[:, [1, 2, 3]]  # Humidity, Light, CO2
        sensor_labels['motion'] = y_onehot
        
        self.data = sensor_data
        self.labels = sensor_labels
        
        return sensor_data, sensor_labels
    
    def _generate_synthetic_data(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """生成合成的智能家居传感器数据"""
        n_samples = 1000
        
        sensor_data = {}
        sensor_labels = {}
        
        # 生成四种传感器数据
        sensor_types = ['temperature', 'humidity', 'light', 'motion']
        
        for sensor_type in sensor_types:
            # 正常数据（占用状态）
            X_occupied = self.rng.normal(loc=25, scale=3, size=(int(n_samples * 0.6), 3))
            
            # 异常数据（未占用状态）
            X_empty = self.rng.normal(loc=20, scale=2, size=(int(n_samples * 0.4), 3))
            
            # 合并
            X = np.vstack([X_occupied, X_empty])
            
            # 标签
            y = np.zeros((n_samples, 2))
            y[:int(n_samples * 0.6), 1] = 1  # 占用
            y[int(n_samples * 0.6):, 0] = 1  # 未占用
            
            # 打乱
            indices = self.rng.permutation(n_samples)
            X = X[indices]
            y = y[indices]
            
            # 归一化
            if self.config.normalize:
                X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
            
            sensor_data[sensor_type] = X
            sensor_labels[sensor_type] = y
        
        self.data = sensor_data
        self.labels = sensor_labels
        
        return sensor_data, sensor_labels


def load_dataset(config: Optional[DatasetConfig] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """加载数据集的便捷函数
    
    Args:
        config: 数据集配置
        
    Returns:
        (data, labels): 传感器数据和标签
    """
    config = config or DatasetConfig()
    
    if config.name.lower() == "skab":
        dataset = SKABDataset(config)
    elif config.name.lower() == "occupancy":
        dataset = OccupancyDataset(config)
    else:
        raise ValueError(f"未知数据集: {config.name}")
    
    return dataset.load()
