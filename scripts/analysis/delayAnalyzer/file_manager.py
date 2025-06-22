"""
文件输出模块
负责保存分析结果到文件
"""
import os
import pandas as pd
import json
import pickle
from datetime import datetime


class FileManager:
    """文件管理器"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.create_directory_structure()
    
    def create_directory_structure(self):
        """创建输出目录结构"""
        subdirs = ['stats', 'plots', 'gmm_params', 'raw_data', 'reports']
        
        os.makedirs(self.output_dir, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
    
    def save_stats_summary(self, stats_dict, load_level=None):
        """保存统计摘要"""
        if not stats_dict:
            return
        
        stats_df = pd.DataFrame(stats_dict).T
        filename = f'delay_stats_summary_{load_level}.csv' if load_level else 'delay_stats_summary.csv'
        filepath = os.path.join(self.output_dir, 'stats', filename)
        
        stats_df.to_csv(filepath, encoding='utf-8-sig')
        print(f"统计摘要已保存到: {filepath}")
        return filepath
    
    def save_fit_quality(self, fit_quality, config, load_level=None):
        """保存拟合质量信息"""
        filename = f'fit_quality_{load_level}.txt' if load_level else 'fit_quality.txt'
        filepath = os.path.join(self.output_dir, 'stats', filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"拟合质量报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")
            f.write(f"负载等级: {load_level or 'N/A'}\n")
            f.write(f"拟合方法: {config.fit_method}\n")
            f.write(f"拟合段数: {config.piece_num if config.piecewise_fit else 1} ({'分段' if config.piecewise_fit else '全局'}拟合)\n")
            f.write(f"均方根误差 (RMSE): {fit_quality.get('rmse', 'N/A'):.6f} us\n")
            f.write(f"平均绝对误差 (MAE): {fit_quality.get('mae', 'N/A'):.6f} us\n")
            f.write(f"决定系数 (R²): {fit_quality.get('r2', 'N/A'):.6f}\n")
            f.write(f"均方误差 (MSE): {fit_quality.get('mse', 'N/A'):.6f}\n")
            f.write(f"数据过滤区间: {config.lower_percent} ~ {config.upper_percent}\n")
            f.write(f"拟合数据区间: {config.fit_lower_percent} ~ {config.fit_upper_percent}\n")
        
        print(f"拟合质量信息已保存到: {filepath}")
        return filepath
    
    def save_gmm_parameters(self, gmm_params, data_type, load_level=None):
        """保存GMM参数"""
        if gmm_params is None:
            return
        
        # 保存为JSON
        filename = f'gmm_params_{data_type}_{load_level}.json' if load_level else f'gmm_params_{data_type}.json'
        json_filepath = os.path.join(self.output_dir, 'gmm_params', filename)
        
        # 确保所有参数都是可序列化的
        serializable_params = {}
        for key, value in gmm_params.items():
            if isinstance(value, (list, tuple)):
                serializable_params[key] = list(value)
            elif hasattr(value, 'item'):  # numpy类型
                serializable_params[key] = value.item()
            else:
                serializable_params[key] = value
        
        with open(json_filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_params, f, indent=2, ensure_ascii=False)
        
        # 也保存为pickle以便后续加载使用
        pickle_filename = filename.replace('.json', '.pkl')
        pickle_filepath = os.path.join(self.output_dir, 'gmm_params', pickle_filename)
        
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(gmm_params, f)
        
        print(f"GMM参数已保存到: {json_filepath} 和 {pickle_filepath}")
        return json_filepath, pickle_filepath
    
    def save_delay_data(self, up_delay, down_delay, load_level=None):
        """保存时延数据"""
        df = pd.DataFrame({
            '自估上行时延': up_delay,
            '自估下行时延': down_delay
        })
        
        filename = f'estimated_delays_{load_level}.csv' if load_level else 'estimated_delays.csv'
        filepath = os.path.join(self.output_dir, 'raw_data', filename)
        
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"时延数据已保存到: {filepath}")
        return filepath
    
    def save_comprehensive_report(self, results_dict, config):
        """保存综合分析报告"""
        report_path = os.path.join(self.output_dir, 'reports', 'comprehensive_report.json')
        
        # 准备报告数据
        report_data = {
            'analysis_timestamp': datetime.now().isoformat(),
            'config': {
                'fit_method': config.fit_method,
                'piecewise_fit': config.piecewise_fit,
                'piece_num': config.piece_num,
                'enable_gmm': config.enable_gmm,
                'gmm_log_transform': config.gmm_log_transform,
                'data_filter_range': [config.lower_percent, config.upper_percent]
            },
            'results': results_dict
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"综合报告已保存到: {report_path}")
        return report_path
    
    def load_gmm_parameters(self, data_type, load_level=None):
        """加载GMM参数"""
        filename = f'gmm_params_{data_type}_{load_level}.pkl' if load_level else f'gmm_params_{data_type}.pkl'
        filepath = os.path.join(self.output_dir, 'gmm_params', filename)
        
        if not os.path.exists(filepath):
            return None
        
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"加载GMM参数失败: {e}")
            return None
    
    def get_output_path(self, subdir, filename):
        """获取输出文件路径"""
        return os.path.join(self.output_dir, subdir, filename)
    
    def create_load_level_dir(self, load_level):
        """为特定负载等级创建目录"""
        load_dir = os.path.join(self.output_dir, f'load_{load_level}')
        subdirs = ['stats', 'plots', 'gmm_params', 'raw_data']
        
        os.makedirs(load_dir, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(load_dir, subdir), exist_ok=True)
        
        return load_dir
