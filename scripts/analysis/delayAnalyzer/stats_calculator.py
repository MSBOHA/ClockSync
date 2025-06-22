"""
统计分析模块
负责计算各种统计量
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class StatisticsCalculator:
    """统计量计算器"""
    
    def calculate_delay_stats(self, up_delay, down_delay):
        """计算时延统计量"""
        stats_dict = {}
        
        for arr, name in zip([up_delay, down_delay], ['自估上行时延', '自估下行时延']):
            arr_clean = pd.Series(arr).dropna()
            if len(arr_clean) == 0:
                continue
                
            stats_dict[name] = {
                'count': len(arr_clean),
                'min': arr_clean.min(),
                'max': arr_clean.max(),
                'mean': arr_clean.mean(),
                'std': arr_clean.std(),
                'median': arr_clean.median(),
                'q25': arr_clean.quantile(0.25),
                'q75': arr_clean.quantile(0.75),
                'iqr': arr_clean.quantile(0.75) - arr_clean.quantile(0.25),
                'skew': arr_clean.skew(),
                'kurtosis': arr_clean.kurtosis()
            }
        
        # 上下行均值
        if '自估上行时延' in stats_dict and '自估下行时延' in stats_dict:
            stats_dict['上下行均值'] = {
                k: (stats_dict['自估上行时延'][k] + stats_dict['自估下行时延'][k]) / 2 
                for k in ['min', 'max', 'mean', 'std', 'median', 'q25', 'q75', 'iqr']
            }
            stats_dict['上下行均值']['count'] = min(stats_dict['自估上行时延']['count'], 
                                               stats_dict['自估下行时延']['count'])
        
        return stats_dict
    
    def calculate_fit_quality(self, y_true, y_pred):
        """计算拟合质量指标"""
        if len(y_true) != len(y_pred) or len(y_true) == 0:
            return {'mse': np.nan, 'rmse': np.nan, 'r2': np.nan, 'mae': np.nan}
        
        # 移除NaN值
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() < 2:
            return {'mse': np.nan, 'rmse': np.nan, 'r2': np.nan, 'mae': np.nan}
        
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        try:
            mse = mean_squared_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true_clean, y_pred_clean)
            mae = np.mean(np.abs(y_true_clean - y_pred_clean))
            
            return {'mse': mse, 'rmse': rmse, 'r2': r2, 'mae': mae}
        except Exception as e:
            print(f"拟合质量计算失败: {e}")
            return {'mse': np.nan, 'rmse': np.nan, 'r2': np.nan, 'mae': np.nan}
    
    def calculate_percentiles(self, data, percentiles=[50, 90, 95, 99]):
        """计算百分位数"""
        data_clean = pd.Series(data).dropna()
        if len(data_clean) == 0:
            return {f'p{p}': np.nan for p in percentiles}
        
        return {f'p{p}': data_clean.quantile(p/100) for p in percentiles}
    
    def get_basic_stats(self, data):
        """获取基本统计量"""
        data_clean = pd.Series(data).dropna()
        if len(data_clean) == 0:
            return {k: np.nan for k in ['count', 'mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis']}
        
        return {
            'count': len(data_clean),
            'mean': data_clean.mean(),
            'std': data_clean.std(),
            'min': data_clean.min(),
            'max': data_clean.max(),
            'median': data_clean.median(),
            'skew': data_clean.skew() if len(data_clean) > 2 else np.nan,
            'kurtosis': data_clean.kurtosis() if len(data_clean) > 3 else np.nan
        }
    
    def calculate_distribution_metrics(self, data):
        """计算分布相关指标"""
        data_clean = pd.Series(data).dropna()
        if len(data_clean) < 4:
            return {}
        
        # 计算变异系数
        cv = data_clean.std() / data_clean.mean() if data_clean.mean() != 0 else np.nan
        
        # 计算四分位距系数
        iqr = data_clean.quantile(0.75) - data_clean.quantile(0.25)
        iqr_coeff = iqr / data_clean.median() if data_clean.median() != 0 else np.nan
        
        # 计算偏度和峰度的标准误差（近似）
        n = len(data_clean)
        skew_se = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))) if n > 3 else np.nan
        kurt_se = np.sqrt(24 * n * (n - 1)**2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5))) if n > 5 else np.nan
        
        return {
            'cv': cv,
            'iqr_coeff': iqr_coeff,
            'skew_se': skew_se,
            'kurt_se': kurt_se,
            'range': data_clean.max() - data_clean.min(),
            'mad': data_clean.mad()  # 平均绝对偏差
        }
