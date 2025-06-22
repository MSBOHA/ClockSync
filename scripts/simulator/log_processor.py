#!/usr/bin/env python3
"""
日志处理工具 - 应用温度和负载影响
输入包含t1,t2,t3,t4的日志，输出受温度和负载影响后的修改版日志
"""
import numpy as np
import pandas as pd
import json
import pickle
import os
from typing import Dict, List, Tuple, Optional
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogProcessor:
    """日志处理器 - 应用温度和负载影响"""
    
    def __init__(self, gmm_params_dir: str = "../../results/load_analysis_lsq/gmm_params"):
        """
        初始化日志处理器
        
        Args:
            gmm_params_dir: GMM参数文件目录
        """
        self.gmm_params_dir = gmm_params_dir
        self.gmm_models = {}
          # 硬编码温度系数 - 基于最新拟合结果 (2025-06-22)
        self.temp_coeffs = {
            'f01': 1.032857,    # 设备100基准频率
            'b1': -1.09e-08,    # 设备100二次系数
            'T01': 47.16,       # 设备100基准温度
            'f02': 1.032857,    # 设备101基准频率  
            'b2': -4.62e-09,    # 设备101二次系数
            'T02': 61.89,       # 设备101基准温度
        }
          # CPU负载映射：cpu1,2,3,4 对应 25%,50%,75%,100%
        self.cpu_level_mapping = {1: 25, 2: 50, 3: 75, 4: 100}
        self.cpu_breakpoints = [25, 50, 75, 100]  # 断点
        
        # 网络负载映射和断点
        self.network_breakpoints = [100, 200, 300, 400, 500, 600]  # Mbps断点
        
        # 统一基准：CPU25%（cpu1）作为所有负载的基准
        self.baseline_samples_cache = {}  # 缓存基准值
        
        logger.info("初始化日志处理器")
        logger.info(f"温度系数: {self.temp_coeffs}")
        logger.info(f"CPU负载映射: {self.cpu_level_mapping}")
        logger.info(f"网络负载断点: {self.network_breakpoints}")
        
    def load_gmm_models(self):
        """加载GMM模型参数"""
        logger.info("开始加载GMM模型...")
        
        # CPU负载等级
        cpu_levels = [1, 2, 3, 4]
        # 网络负载等级 (Mbps)
        network_levels = [100, 200, 300, 400, 500, 600]
        
        for cpu_level in cpu_levels:
            for direction in ['上行', '下行']:
                key = f"cpu_{cpu_level}_{direction}"
                pkl_file = os.path.join(self.gmm_params_dir, f"gmm_params_{direction}delay_{cpu_level}.pkl")
                json_file = os.path.join(self.gmm_params_dir, f"gmm_params_{direction}delay_{cpu_level}.json")
                
                if os.path.exists(pkl_file):
                    try:
                        with open(pkl_file, 'rb') as f:
                            self.gmm_models[key] = pickle.load(f)
                        logger.info(f"加载GMM模型: {key}")
                    except Exception as e:
                        logger.warning(f"加载GMM模型失败 {key}: {e}")
                elif os.path.exists(json_file):
                    try:
                        with open(json_file, 'r') as f:
                            self.gmm_models[key] = json.load(f)
                        logger.info(f"加载GMM参数: {key}")
                    except Exception as e:
                        logger.warning(f"加载GMM参数失败 {key}: {e}")
        
        for network_level in network_levels:
            for direction in ['上行', '下行']:
                key = f"network_{network_level}_{direction}"
                pkl_file = os.path.join(self.gmm_params_dir, f"gmm_params_{direction}delay_{network_level}.pkl")
                json_file = os.path.join(self.gmm_params_dir, f"gmm_params_{direction}delay_{network_level}.json")
                
                if os.path.exists(pkl_file):
                    try:
                        with open(pkl_file, 'rb') as f:
                            self.gmm_models[key] = pickle.load(f)
                        logger.info(f"加载GMM模型: {key}")
                    except Exception as e:
                        logger.warning(f"加载GMM模型失败 {key}: {e}")
                elif os.path.exists(json_file):
                    try:
                        with open(json_file, 'r') as f:
                            self.gmm_models[key] = json.load(f)
                        logger.info(f"加载GMM参数: {key}")
                    except Exception as e:
                        logger.warning(f"加载GMM参数失败 {key}: {e}")
                        
        logger.info(f"成功加载 {len(self.gmm_models)} 个GMM模型")
        
    def sample_delay_from_gmm(self, load_type: str, load_level: int, direction: str, n_samples: int = 1) -> np.ndarray:
        """
        从GMM模型采样时延
        
        Args:
            load_type: 负载类型 ('cpu' 或 'network')
            load_level: 负载等级
            direction: 方向 ('上行' 或 '下行')
            n_samples: 采样数量
            
        Returns:
            采样的时延值 (微秒)
        """
        key = f"{load_type}_{load_level}_{direction}"
        
        if key not in self.gmm_models:
            logger.warning(f"未找到GMM模型: {key}，使用默认值")
            return np.random.normal(0, 1e-6, n_samples) * 1e6  # 默认微秒级噪声
            
        gmm_params = self.gmm_models[key]
          # 如果是pickle加载的sklearn模型
        if hasattr(gmm_params, 'sample'):
            try:
                samples, _ = gmm_params.sample(n_samples)
                # 对数正态分布：需要指数变换
                delay_samples = np.exp(samples.flatten())
                logger.debug(f"GMM采样 {key}: log空间 {samples.flatten()[:3]} -> 时延空间 {delay_samples[:3]} μs")
                return delay_samples
            except Exception as e:
                logger.warning(f"GMM采样失败 {key}: {e}")
                return np.exp(np.random.normal(-10, 1, n_samples))  # 默认对数正态噪声
        
        # 如果是JSON参数，手动实现GMM采样
        elif isinstance(gmm_params, dict):
            try:
                weights = np.array(gmm_params['weights'])
                means = np.array(gmm_params['means'])
                covariances = np.array(gmm_params['covariances'])
                
                # 根据权重选择高斯分量
                component_indices = np.random.choice(len(weights), size=n_samples, p=weights)
                
                # 从选中的高斯分量采样（对数空间）
                log_samples = []
                for idx in component_indices:
                    log_sample = np.random.normal(means[idx], np.sqrt(covariances[idx]))
                    log_samples.append(log_sample)
                
                log_samples = np.array(log_samples)
                # 对数正态分布：指数变换得到真实时延
                delay_samples = np.exp(log_samples)
                logger.debug(f"手动GMM采样 {key}: log空间 {log_samples[:3]} -> 时延空间 {delay_samples[:3]} μs")
                return delay_samples
            except Exception as e:
                logger.warning(f"手动GMM采样失败 {key}: {e}")
                return np.exp(np.random.normal(-10, 1, n_samples))  # 默认对数正态噪声        else:
            logger.warning(f"未知的GMM模型格式: {key}")
            return np.exp(np.random.normal(-10, 1, n_samples))  # 默认对数正态噪声
    
    def calculate_temperature_effect(self, temp_100: float, temp_101: float) -> float:
        """
        计算温度对频率的影响 (二次模型)
        
        Args:
            temp_100: 设备100温度 (°C)
            temp_101: 设备101温度 (°C)
            
        Returns:
            频率比 f100/f101
        """
        # 二次模型: f1/f2 = [f01*(1-b1*(T1-T01)^2)] / [f02*(1-b2*(T2-T02)^2)]
        f1 = self.temp_coeffs['f01'] * (1 - self.temp_coeffs['b1'] * (temp_100 - self.temp_coeffs['T01'])**2)
        f2 = self.temp_coeffs['f02'] * (1 - self.temp_coeffs['b2'] * (temp_101 - self.temp_coeffs['T02'])**2)        
        return f1 / f2
    
    def process_log_file(self, input_file: str, output_file: str, 
                        cpu_load: float = 25.0, network_load: int = 100,
                        temp_100: float = 35.0, temp_101: float = 55.0,
                        apply_temperature: bool = True, apply_load: bool = True):
        """
        处理日志文件，应用温度和负载影响
        
        Args:
            input_file: 输入日志文件
            output_file: 输出日志文件
            cpu_load: CPU负载百分比 (0-100)
            network_load: 网络负载等级 (100-600 Mbps)
            temp_100: 设备100温度 (°C)
            temp_101: 设备101温度 (°C)
            apply_temperature: 是否应用温度影响            apply_load: 是否应用负载影响
        """
        logger.info(f"开始处理日志文件: {input_file}")
        logger.info(f"配置: CPU={cpu_load}%, 网络={network_load}Mbps, T100={temp_100}°C, T101={temp_101}°C")
        
        # 读取原始日志
        data = self.load_log_file(input_file)
        if data is None:
            logger.error("日志文件读取失败")
            return
            
        n_lines = len(data)
        logger.info(f"读取到 {n_lines} 行数据")
        
        # 提取t1, t2, t3, t4
        t1_orig = data[:, 0]
        t2_orig = data[:, 1] 
        t3_orig = data[:, 2]
        t4_orig = data[:, 3]
        
        # 初始化修改后的时间戳
        t1_mod = t1_orig.copy()
        t2_mod = t2_orig.copy()
        t3_mod = t3_orig.copy()
        t4_mod = t4_orig.copy()        # 应用温度影响 - 只影响t2和t3
        if apply_temperature:
            logger.info("应用温度影响...")
            freq_ratio = self.calculate_temperature_effect(temp_100, temp_101)
            logger.info(f"频率比 f100/f101 = {freq_ratio:.8f}")
            
            # 计算相对于上一个时间点的deltaT（增量时间）
            # 对每一行计算温度漂移
            for i in range(n_lines):
                if i == 0:
                    delta_t = 0  # 第一个点没有漂移
                else:
                    # 使用相邻时间点的差值，而不是累积时间
                    delta_t = t1_orig[i] - t1_orig[i-1]  # 增量时间
                
                # 只影响t2和t3 (设备101的时间戳)
                # 频率漂移是增量的，不是累积的
                freq_drift = delta_t * (freq_ratio - 1.0)
                
                if i == 0:
                    t2_mod[i] = t2_orig[i]  # 第一个点不变
                    t3_mod[i] = t3_orig[i]
                else:
                    t2_mod[i] = t2_mod[i-1] + (t2_orig[i] - t2_orig[i-1]) + freq_drift
                    t3_mod[i] = t3_mod[i-1] + (t3_orig[i] - t3_orig[i-1]) + freq_drift        # 应用负载影响
        if apply_load:
            logger.info("应用负载影响...")
            
            # 采样负载的增量影响（相对于基准负载）
            uplink_impact_cpu = self.sample_load_impact('cpu', cpu_load, '上行', n_lines)
            downlink_impact_cpu = self.sample_load_impact('cpu', cpu_load, '下行', n_lines)
            
            # 网络负载增量影响
            uplink_impact_net = self.sample_load_impact('network', network_load, '上行', n_lines)
            downlink_impact_net = self.sample_load_impact('network', network_load, '下行', n_lines)
              # 合成总增量时延 (微秒转秒)
            dsm = (uplink_impact_cpu + uplink_impact_net) * 1e-6  # 上行增量时延
            dms = (downlink_impact_cpu + downlink_impact_net) * 1e-6  # 下行增量时延
            
            # 输出详细统计信息
            logger.info(self.format_stats(uplink_impact_cpu, f"CPU {cpu_load}%负载增量-上行"))
            logger.info(self.format_stats(downlink_impact_cpu, f"CPU {cpu_load}%负载增量-下行"))
            logger.info(self.format_stats(uplink_impact_net, f"网络 {network_load}Mbps负载增量-上行"))
            logger.info(self.format_stats(downlink_impact_net, f"网络 {network_load}Mbps负载增量-下行"))
            logger.info(self.format_stats(dsm*1e6, f"总上行增量dsm"))
            logger.info(self.format_stats(dms*1e6, f"总下行增量dms"))
            
            # 按照您的要求应用时延: t1, t2+dsm, t3+dsm, t4+dsm+dms
            # t1保持不变
            t2_mod += dsm     # t2增加上行时延
            t3_mod += dsm     # t3增加上行时延  
            t4_mod += dsm + dms  # t4增加上行+下行时延
        
        # 构建输出数据
        output_data = data.copy()
        output_data[:, 0] = t1_mod
        output_data[:, 1] = t2_mod
        output_data[:, 2] = t3_mod
        output_data[:, 3] = t4_mod
        
        # 写入输出文件
        self.save_log_file(output_data, output_file)
        
        # 统计信息        logger.info("处理完成，统计信息:")
        t1_change = (t1_mod - t1_orig)*1e6
        t2_change = (t2_mod - t2_orig)*1e6
        t3_change = (t3_mod - t3_orig)*1e6
        t4_change = (t4_mod - t4_orig)*1e6
        
        logger.info(self.format_stats(t1_change, "t1变化"))
        logger.info(self.format_stats(t2_change, "t2变化"))
        logger.info(self.format_stats(t3_change, "t3变化"))
        logger.info(self.format_stats(t4_change, "t4变化"))
        logger.info(f"输出文件: {output_file}")
    
    def load_log_file(self, filepath: str) -> Optional[np.ndarray]:
        """
        加载日志文件 (支持多种格式)
        
        Args:
            filepath: 日志文件路径
            
        Returns:
            数据数组，前4列为t1,t2,t3,t4
        """
        try:
            # 尝试直接读取为数值
            data = np.loadtxt(filepath)
            if data.shape[1] >= 4:
                return data
            else:
                logger.error(f"日志文件列数不足: {data.shape[1]} < 4")
                return None
                
        except Exception as e:
            logger.warning(f"数值读取失败，尝试文本解析: {e}")
            
            # 尝试文本解析
            try:
                with open(filepath, 'r') as f:
                    lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('//')]
                
                data = []
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            # 尝试提取前4个数值
                            row = [float(parts[i]) for i in range(min(len(parts), 10))]  # 最多10列
                            data.append(row)
                        except ValueError:
                            continue
                
                if len(data) > 0:
                    data = np.array(data)
                    logger.info(f"文本解析成功: {data.shape}")
                    return data
                else:
                    logger.error("文本解析失败，无有效数据")
                    return None
                    
            except Exception as e2:
                logger.error(f"文本解析也失败: {e2}")
                return None
    
    def save_log_file(self, data: np.ndarray, filepath: str):
        """
        保存日志文件
        
        Args:
            data: 数据数组
            filepath: 输出文件路径
        """
        try:
            # 保存为空格分隔的文本文件
            np.savetxt(filepath, data, fmt='%.9f', delimiter=' ')
            logger.info(f"保存成功: {filepath} ({data.shape[0]} 行)")
        except Exception as e:
            logger.error(f"保存失败: {e}")
    
    def sample_cpu_delay_interpolated(self, cpu_percent: float, direction: str, n_samples: int = 1) -> np.ndarray:
        """
        根据CPU负载百分比进行插值采样时延
        
        Args:
            cpu_percent: CPU负载百分比 (0-100)
            direction: 方向 ('上行' 或 '下行')
            n_samples: 采样数量
            
        Returns:
            采样的时延值 (微秒)
        """
        # 限制范围
        cpu_percent = max(0, min(100, cpu_percent))
        
        # 如果 <= 25%，直接从25%采样
        if cpu_percent <= 25:
            return self.sample_delay_from_gmm('cpu', 1, direction, n_samples)
        
        # 找到最近的两个断点
        for i in range(len(self.cpu_breakpoints) - 1):
            if cpu_percent <= self.cpu_breakpoints[i + 1]:
                lower_percent = self.cpu_breakpoints[i]
                upper_percent = self.cpu_breakpoints[i + 1]
                
                # 对应的CPU等级
                lower_level = i + 1  # cpu1,2,3,4
                upper_level = i + 2
                
                # 线性插值权重
                weight_upper = (cpu_percent - lower_percent) / (upper_percent - lower_percent)
                weight_lower = 1 - weight_upper
                
                logger.debug(f"CPU {cpu_percent}% 插值: {lower_percent}%({weight_lower:.3f}) + {upper_percent}%({weight_upper:.3f})")
                
                # 从两个等级采样
                samples_lower = self.sample_delay_from_gmm('cpu', lower_level, direction, n_samples)
                samples_upper = self.sample_delay_from_gmm('cpu', upper_level, direction, n_samples)
                
                # 线性插值合成
                interpolated_samples = weight_lower * samples_lower + weight_upper * samples_upper
                return interpolated_samples
                  # 如果超过100%，直接从100%采样
        return self.sample_delay_from_gmm('cpu', 4, direction, n_samples)
    
    def sample_load_impact(self, load_type: str, load_level: float, direction: str, n_samples: int = 1) -> np.ndarray:
        """
        采样负载的增量影响（相对于统一基准CPU25%）
        
        Args:
            load_type: 负载类型 ('cpu' 或 'network')
            load_level: 负载等级 (CPU: 0-100%, Network: 100-600Mbps)
            direction: 方向 ('上行' 或 '下行')
            n_samples: 采样数量
            
        Returns:
            负载增量时延 (微秒)
        """        # 统一基准：CPU25%的固定基准值
        baseline_delay = self.get_baseline_delay(direction)
        
        if load_type == 'cpu':
            if load_level <= 25:
                # 小于等于基准负载，按比例缩放CPU25%的时延
                if load_level <= 0:
                    # 0%CPU负载，无时延增量
                    return np.zeros(n_samples)
                else:
                    # 按比例缩放：load_level/25
                    base_samples = self.sample_delay_from_gmm('cpu', 1, direction, n_samples)
                    scale_factor = load_level / 25.0
                    load_samples = base_samples * scale_factor
                  # 计算增量并截断负值
                impact = load_samples - baseline_delay
                impact = np.maximum(impact, 0)
                return impact
            else:
                # 使用插值采样获取负载下的时延
                load_samples = self.sample_cpu_delay_interpolated(load_level, direction, n_samples)
                # 返回增量：负载时延 - 基准时延
                impact = load_samples - baseline_delay
                # 负值截断：如果小于0就取0（无负载影响）
                impact = np.maximum(impact, 0)
                return impact
                
        elif load_type == 'network':
            # 获取网络基准时延（网络100Mbps）
            network_baseline_delay = self.get_network_baseline_delay(direction)
            
            # 使用插值采样网络负载时延
            load_samples = self.sample_network_delay_interpolated(load_level, direction, n_samples)
            # 返回增量：网络负载时延 - 网络基准时延（100Mbps）
            impact = load_samples - network_baseline_delay            # 负值截断：如果小于0就取0（无负载影响）
            impact = np.maximum(impact, 0)
            return impact
        else:
            logger.warning(f"未知负载类型: {load_type}")
            return np.zeros(n_samples)
    def get_baseline_delay(self, direction: str) -> float:
        """
        获取基准时延值（CPU25%作为基准）
        
        Args:
            direction: 方向 ('上行' 或 '下行')
            
        Returns:
            基准时延值 (微秒)
        """
        key = f"baseline_{direction}"
        if key not in self.baseline_samples_cache:
            # 从CPU25%（cpu1）采样5000次，取中位数作为基准，保留合理的波动特征
            samples = self.sample_delay_from_gmm('cpu', 1, direction, 5000)
            baseline_value = np.median(samples)  # 使用中位数作为基准
            
            self.baseline_samples_cache[key] = baseline_value
            logger.info(f"计算CPU基准时延 {direction}: {baseline_value:.2f} μs (基于CPU25%，5000次采样的中位数)")
        
        return self.baseline_samples_cache[key]
    
    def get_network_baseline_delay(self, direction: str) -> float:
        """
        获取网络基准时延值（网络100Mbps作为基准）
        
        Args:
            direction: 方向 ('上行' 或 '下行')
            
        Returns:
            网络基准时延值 (微秒)
        """
        key = f"network_baseline_{direction}"
        if key not in self.baseline_samples_cache:
            # 从网络100Mbps采样5000次，取中位数作为网络基准
            samples = self.sample_delay_from_gmm('network', 100, direction, 5000)
            baseline_value = np.median(samples)  # 使用中位数作为基准
            
            self.baseline_samples_cache[key] = baseline_value
            logger.info(f"计算网络基准时延 {direction}: {baseline_value:.2f} μs (基于网络100Mbps，5000次采样的中位数)")
        
        return self.baseline_samples_cache[key]
    
    def sample_network_delay_interpolated(self, network_mbps: float, direction: str, n_samples: int = 1) -> np.ndarray:
        """
        根据网络负载Mbps进行插值采样时延
        
        Args:
            network_mbps: 网络负载 (Mbps)
            direction: 方向 ('上行' 或 '下行')
            n_samples: 采样数量
            
        Returns:
            采样的时延值 (微秒)
        """
        # 处理小于100Mbps的情况：按比例缩放100Mbps的采样值
        if network_mbps <= 100:
            base_samples = self.sample_delay_from_gmm('network', 100, direction, n_samples)
            if network_mbps <= 0:

                # 0负载，无时延增量
                return np.zeros(n_samples)
            else:
                # 按比例缩放：network_mbps/100
                scale_factor = network_mbps / 100.0
                return base_samples * scale_factor
        
        # 限制上限范围
        network_mbps = min(600, network_mbps)
        
        # 如果正好在断点上，直接采样
        if network_mbps in self.network_breakpoints:
            return self.sample_delay_from_gmm('network', int(network_mbps), direction, n_samples)
        
        # 找到最近的两个断点
        for i in range(len(self.network_breakpoints) - 1):
            if network_mbps <= self.network_breakpoints[i + 1]:
                lower_mbps = self.network_breakpoints[i]
                upper_mbps = self.network_breakpoints[i + 1]
                
                # 线性插值权重
                weight_upper = (network_mbps - lower_mbps) / (upper_mbps - lower_mbps)
                weight_lower = 1 - weight_upper
                
                logger.debug(f"网络 {network_mbps}Mbps 插值: {lower_mbps}Mbps({weight_lower:.3f}) + {upper_mbps}Mbps({weight_upper:.3f})")
                
                # 从两个负载点采样
                samples_lower = self.sample_delay_from_gmm('network', int(lower_mbps), direction, n_samples)
                samples_upper = self.sample_delay_from_gmm('network', int(upper_mbps), direction, n_samples)
                
                # 线性插值合成
                interpolated_samples = weight_lower * samples_lower + weight_upper * samples_upper
                return interpolated_samples
                
        # 超出范围，使用边界值
        if network_mbps <= 100:
            return self.sample_delay_from_gmm('network', 100, direction, n_samples)
        else:
            return self.sample_delay_from_gmm('network', 600, direction, n_samples)
    
    def format_stats(self, data: np.ndarray, name: str, unit: str = "μs") -> str:
        """
        格式化统计信息输出
        
        Args:
            data: 数据数组
            name: 统计量名称
            unit: 单位
            
        Returns:
            格式化的统计信息字符串
        """
        mean_val = np.mean(data)
        min_val = np.min(data)
        max_val = np.max(data)
        std_val = np.std(data)
        median_val = np.median(data)
        q25 = np.percentile(data, 25)
        q75 = np.percentile(data, 75)
        
        return (f"{name}: 均值={mean_val:.2f}{unit}, 中位数={median_val:.2f}{unit}, "
                f"范围=[{min_val:.2f}, {max_val:.2f}]{unit}, σ={std_val:.2f}{unit}, "
                f"四分位=[{q25:.2f}, {q75:.2f}]{unit}")

def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='日志处理工具 - 应用温度和负载影响')
    parser.add_argument('input_file', help='输入日志文件路径')
    parser.add_argument('output_file', help='输出日志文件路径')
    parser.add_argument('--cpu-load', type=float, default=25.0,
                       help='CPU负载百分比 (0-100)')
    parser.add_argument('--network-load', type=float, default=100.0,
                       help='网络负载 (100-600 Mbps)')
    parser.add_argument('--temp-100', type=float, default=35.0,
                       help='设备100温度 (°C)')
    parser.add_argument('--temp-101', type=float, default=55.0,
                       help='设备101温度 (°C)')
    parser.add_argument('--no-temperature', action='store_true',
                       help='不应用温度影响')
    parser.add_argument('--no-load', action='store_true',
                       help='不应用负载影响')
    parser.add_argument('--gmm-dir', type=str, 
                       default='../../results/load_analysis_lsq/gmm_params',
                       help='GMM参数目录')
    
    args = parser.parse_args()
    
    # 创建日志处理器
    processor = LogProcessor(gmm_params_dir=args.gmm_dir)
    
    # 加载GMM模型
    processor.load_gmm_models()
    
    # 处理日志文件
    processor.process_log_file(
        input_file=args.input_file,
        output_file=args.output_file,
        cpu_load=args.cpu_load,
        network_load=args.network_load,
        temp_100=args.temp_100,
        temp_101=args.temp_101,
        apply_temperature=not args.no_temperature,
        apply_load=not args.no_load
    )


if __name__ == "__main__":
    main()
#示例
# python log_processor.py "log0-original.log" "processed_corrected_v2.log" --cpu-load 1 --network-load 400 --temp-100 35.0 --temp-101 50.0