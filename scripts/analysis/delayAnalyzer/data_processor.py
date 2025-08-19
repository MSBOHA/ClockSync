"""
数据处理模块
负责数据加载、清洗和预处理
"""
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """数据处理器"""
    
    @staticmethod
    def get_column_names():
        """获取标准列名"""
        return [
            "ns2s(getT1(REAL))", "ns2s(getT2(REAL))", "ns2s(getT3(REAL))", "ns2s(getT4(REAL))",
            "ns2us(getT2T1(REAL))", "ns2us(getT3T4(REAL))", "ns2us(getOffset(REAL))", "ns2us(getCalculatedOffset(REAL))",
            "ns2us(getDelay(REAL))", "ns2s(getT4T1(REAL))",
            "ns2s(getT1(RAW))", "ns2s(getT2(RAW))", "ns2s(getT3(RAW))", "ns2s(getT4(RAW))",
            "ns2us(getT2T1(RAW))", "ns2us(getT3T4(RAW))", "ns2us(getOffset(RAW))", "ns2us(getCalculatedOffset(RAW))",
            "ns2us(getDelay(RAW))", "ns2s(getT4T1(RAW))",
            "seqNo", "ns2us(t1RealKernel - t1RealApp)", "ns2us(t2RealApp - t2RealKernel)", "ns2us(t3RealKernel - t3RealApp)",
            "ns2us(t4RealApp - t4RealKernel)", "ns2us(models[RAW1.local.calculate0ffsetToRemote(getT1(RAW))])",
            "ns2us(getT2T1(RAW)-models[RAW1.local.calculate0ffsetToRemote(getT1(RAW))])",
            "ns2us(models[RAW1.local.calculate0ffsetToRemote(getT1(RAW))-getT3T4(RAW)])"
        ]
    
    def load_data(self, log_file):
        """加载数据文件"""
        try:
            column_names = self.get_column_names()
            df = pd.read_csv(log_file, sep='\s+', header=None)
            
            if len(df.columns) >= len(column_names):
                df.columns = column_names + [f"col_{i}" for i in range(len(df.columns) - len(column_names))]
            else:
                df.columns = column_names[:len(df.columns)]
            
            print(f"成功加载数据: {len(df)} 行, {len(df.columns)} 列")
            return df
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def filter_outliers(self, df, lower_percent=0.01, upper_percent=0.99):
        """基于分位数过滤异常值"""
        # 检查是否有足够的列数
        if len(df.columns) < 19:
            print(f"警告: 数据列数不足 ({len(df.columns)} < 19)，跳过过滤")
            return df
        
        # 检查是否存在目标列
        delay_col = "ns2us(getDelay(RAW))"
        if delay_col not in df.columns:
            print(f"警告: 未找到列 '{delay_col}'，可用列: {list(df.columns)}")
            return df
        
        delay_raw = df[delay_col]
        q_low = delay_raw.quantile(lower_percent)
        q_high = delay_raw.quantile(upper_percent)
        
        print(f"分位过滤区间: {q_low:.2f} ~ {q_high:.2f} us")
        mask = (delay_raw >= q_low) & (delay_raw <= q_high)
        filtered_df = df[mask].reset_index(drop=True)
        print(f"过滤前: {len(df)} 行, 过滤后: {len(filtered_df)} 行")
        
        return filtered_df
    
    def extract_load_level(self, log_file):
        """从文件名提取负载等级"""
        import re
        # 匹配类似 "400M", "cpu1" 等模式
        patterns = [
            r'(\d+)M',  # 数字+M
            r'cpu(\d+)',  # cpu+数字
            r'temp(\d+)',  # temp+数字
        ]
        
        for pattern in patterns:
            match = re.search(pattern, log_file)
            if match:
                return match.group(1)
        
        # 如果没有匹配到，返回文件名
        import os
        return os.path.basename(log_file).split('.')[0]


class OffsetFitter:
    """Offset拟合器"""
    
    def __init__(self, method='svr'):
        self.method = method
    
    def fit(self, df, fit_lower_percent=0.01, fit_upper_percent=0.3, 
            piecewise=False, piece_num=3):
        """拟合offset"""
        t1_raw = df["ns2s(getT1(RAW))"]
        offset_raw = df["ns2us(getOffset(RAW))"]
        delay_raw = df["ns2us(getDelay(RAW))"]
          # 获取用于拟合的数据子集
        fit_q_low = delay_raw.quantile(fit_lower_percent)
        fit_q_high = delay_raw.quantile(fit_upper_percent)
        fit_mask = (delay_raw >= fit_q_low) & (delay_raw <= fit_q_high)
        
        print(f"  分位过滤区间: {fit_q_low:.2f} ~ {fit_q_high:.2f} us")
        print(f"  过滤前数据点: {len(df)}, 过滤后数据点: {fit_mask.sum()}")
        
        if piecewise and piece_num > 1:
            # 修复bug: 分段拟合也应该使用过滤后的数据
            offset_est = self._piecewise_fit(t1_raw[fit_mask], offset_raw[fit_mask], piece_num, t1_raw)
        else:
            offset_est = self._global_fit(t1_raw[fit_mask], offset_raw[fit_mask], t1_raw)
        return offset_est
    
    def _global_fit(self, X_fit, y_fit, X_all):
        """全局拟合"""
        if len(X_fit) < 2:
            return np.full_like(X_all.values, np.nan)
        
        if self.method == 'svr':
            return self._svr_fit(X_fit.values, y_fit.values, X_all.values)
        else:  # lsq
            return self._lsq_fit(X_fit.values, y_fit.values, X_all.values)
    
    def _piecewise_fit(self, X_fit, y_fit, piece_num, X_all=None):
        """分段拟合"""
        print(f"  执行{piece_num}段拟合...")
        
        # 如果没有提供X_all，使用X_fit
        if X_all is None:
            X_all = X_fit
            
        xvals_fit = X_fit.values
        yvals_fit = y_fit.values
        xvals_all = X_all.values
        offset_est = np.full_like(xvals_all, np.nan, dtype=float)
        
        # 分段边界基于拟合数据
        xs = np.percentile(xvals_fit, np.linspace(0, 100, piece_num + 1))
        xs[-1] = xvals_fit.max()
        
        total_fit_points = 0
        total_pred_points = 0
        
        for i in range(piece_num):
            if i == piece_num - 1:
                seg_mask_fit = (xvals_fit >= xs[i]) & (xvals_fit <= xs[i+1])
                seg_mask_all = (xvals_all >= xs[i]) & (xvals_all <= xs[i+1])
            else:
                seg_mask_fit = (xvals_fit >= xs[i]) & (xvals_fit < xs[i+1])
                seg_mask_all = (xvals_all >= xs[i]) & (xvals_all < xs[i+1])
            
            X_seg = xvals_fit[seg_mask_fit]
            y_seg = yvals_fit[seg_mask_fit]
            X_pred = xvals_all[seg_mask_all]
            
            if len(X_seg) < 2:
                continue
            
            total_fit_points += len(X_seg)
            total_pred_points += len(X_pred)
            
            if self.method == 'svr':
                pred = self._svr_fit(X_seg, y_seg, X_pred)
            else:
                pred = self._lsq_fit(X_seg, y_seg, X_pred)
            
            offset_est[seg_mask_all] = pred
        
        print(f"  拟合完成: 总拟合点数 {total_fit_points}, 总预测点数 {total_pred_points}")
        return offset_est

    def _svr_fit(self, X, y, X_pred):
        """SVR拟合"""
        try:
            X_mean, X_std = X.mean(), X.std()
            y_mean, y_std = y.mean(), y.std()
            
            if X_std == 0 or y_std == 0:
                return np.full_like(X_pred, y_mean)
            
            # 如果数据点太多，进行采样以加速拟合
            if len(X) > 5000:
                indices = np.random.choice(len(X), 5000, replace=False)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample = X
                y_sample = y
            
            X_norm = (X_sample - X_mean) / X_std
            y_norm = (y_sample - y_mean) / y_std
            
            # 使用更快的参数设置
            svr = SVR(kernel='linear', C=10, epsilon=0.1, max_iter=1000)
            svr.fit(X_norm.reshape(-1, 1), y_norm)
            
            X_pred_norm = (X_pred - X_mean) / X_std
            pred_norm = svr.predict(X_pred_norm.reshape(-1, 1))
            
            return pred_norm * y_std + y_mean
        except Exception as e:
            print(f"SVR拟合失败: {e}")
            return np.full_like(X_pred, y.mean())
    
    def _lsq_fit(self, X, y, X_pred):
        """最小二乘拟合"""
        try:
            coef = np.polyfit(X, y, 1)
            return np.polyval(coef, X_pred)
        except Exception as e:
            print(f"最小二乘拟合失败: {e}")
            return np.full_like(X_pred, y.mean())


class GMMFitter:
    """高斯混合模型拟合器"""
    
    def __init__(self, n_components_range=(1, 6), log_transform=True):
        self.n_components_range = n_components_range
        self.log_transform = log_transform
    
    def fit_gmm(self, data, load_level=None):
        """对数据进行GMM拟合"""
        data_clean = pd.Series(data).dropna()
        if len(data_clean) < 10:
            print(f"数据点过少 ({len(data_clean)})，跳过GMM拟合")
            return None
        
        # 数据预处理
        if self.log_transform:
            # 确保数据为正值
            data_clean = data_clean[data_clean > 0]
            if len(data_clean) < 10:
                print("正值数据点过少，跳过GMM拟合")
                return None
            data_transformed = np.log(data_clean.values)
        else:
            data_transformed = data_clean.values
        
        # 选择最优组件数
        best_gmm, best_bic = self._select_best_components(data_transformed)
        
        if best_gmm is None:
            return None
        
        # 提取参数
        gmm_params = self._extract_gmm_parameters(best_gmm, data_transformed)
        gmm_params['load_level'] = load_level
        gmm_params['log_transformed'] = self.log_transform
        gmm_params['n_samples'] = len(data_clean)
        gmm_params['original_mean'] = data_clean.mean()
        gmm_params['original_std'] = data_clean.std()
        
        return gmm_params
    
    def _select_best_components(self, data):
        """选择最优的组件数"""
        best_bic = np.inf
        best_gmm = None
        
        for n_components in range(self.n_components_range[0], self.n_components_range[1] + 1):
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    random_state=42,
                    max_iter=200
                )
                gmm.fit(data.reshape(-1, 1))
                
                bic = gmm.bic(data.reshape(-1, 1))
                
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
                    
            except Exception as e:
                print(f"GMM拟合失败 (n_components={n_components}): {e}")
                continue
        
        return best_gmm, best_bic
    
    def _extract_gmm_parameters(self, gmm, data):
        """提取GMM参数"""
        params = {
            'n_components': gmm.n_components,
            'weights': gmm.weights_.tolist(),
            'means': gmm.means_.flatten().tolist(),
            'covariances': gmm.covariances_.flatten().tolist(),
            'bic': gmm.bic(data.reshape(-1, 1)),
            'aic': gmm.aic(data.reshape(-1, 1)),
            'log_likelihood': gmm.score(data.reshape(-1, 1))
        }
        
        # 计算每个组件的标准差
        params['stds'] = [np.sqrt(cov) for cov in params['covariances']]
        
        return params
