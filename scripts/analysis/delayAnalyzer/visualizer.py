"""
可视化模块
负责生成各种图表
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from scipy.stats import norm


class Visualizer:
    """可视化器"""
    
    def __init__(self):
        self._setup_style()
        self._ensure_chinese_font()  # 确保中文字体设置生效
    def _setup_style(self):
        """设置绘图样式"""
        # 设置中文字体支持 - 使用多个备选字体
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        
        # 清除matplotlib字体缓存，确保新设置生效
        import matplotlib.font_manager as fm
        try:
            fm._get_fontconfig_fonts.cache_clear()
        except:
            pass  # 某些版本可能没有这个方法
        
        sns.set_style('whitegrid')
        sns.set_palette('Set2')
        
        # 验证字体设置
        print(f"✅ 字体设置: {plt.rcParams['font.sans-serif'][:3]}")  # 显示前3个字体
    
    def _ensure_chinese_font(self):
        """确保中文字体设置生效"""
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_raw_time_series(self, df, offset_est, fit_quality, output_dir, 
                           piecewise=False, piece_num=1, fit_method='svr'):
        """绘制原始时间序列图"""
        self._ensure_chinese_font()  # 确保中文字体设置
        
        t1_raw = df["ns2s(getT1(RAW))"]
        t2t1_raw = df["ns2us(getT2T1(RAW))"]
        t3t4_raw = df["ns2us(getT3T4(RAW))"]
        
        plt.figure(figsize=(12, 6))
        plt.plot(t1_raw * 1e6, t2t1_raw, label='T2-T1 (RAW)', alpha=0.7, 
                marker='.', markersize=2, linestyle='-')
        plt.plot(t1_raw * 1e6, t3t4_raw, label='T3-T4 (RAW)', alpha=0.7, 
                marker='.', markersize=2, linestyle='-')
        
        if offset_est is not None and not np.isnan(offset_est).all():
            fit_type = f'{piece_num}段' if piecewise and piece_num > 1 else '全局'
            plt.plot(t1_raw * 1e6, offset_est, label=f'{fit_type}拟合Offset',
                    color='black', linewidth=2.5, alpha=0.8)
            
            # 添加拟合质量信息
            fit_info = (f'拟合方法: {fit_method}\n'
                       f'RMSE: {fit_quality.get("rmse", np.nan):.4f} us\n'
                       f'R²: {fit_quality.get("r2", np.nan):.4f}')
            plt.annotate(fit_info, xy=(0.02, 0.97), xycoords='axes fraction',
                        fontsize=10, bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor="wheat", alpha=0.7),
                        verticalalignment='top')
        
        plt.xlabel('时间 (μs)')
        plt.ylabel('时延 (μs)')
        plt.title('原始时间序列 - T2T1和T3T4')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        output_file = os.path.join(output_dir, 'raw_t2t1_t3t4_offset_vs_t1.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_delay_comparison(self, stats_dict, output_dir):
        """绘制时延统计对比图"""
        self._ensure_chinese_font()  # 确保中文字体设置
        
        if not stats_dict:
            return
        
        # 提取统计数据
        metrics = ['mean', 'std', 'min', 'max']
        delay_types = []
        data = {metric: [] for metric in metrics}
        
        for delay_type, stats in stats_dict.items():
            if isinstance(stats, dict) and 'mean' in stats:
                delay_types.append(delay_type)
                for metric in metrics:
                    data[metric].append(stats.get(metric, 0))
        
        if not delay_types:
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            bars = ax.bar(delay_types, data[metric], alpha=0.7)
            ax.set_title(f'时延{metric.upper()}对比')
            ax.set_ylabel(f'{metric.upper()} (μs)')
            
            # 添加数值标签
            for j, bar in enumerate(bars):
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            # 旋转x轴标签以避免重叠
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 保存图片
        output_file = os.path.join(output_dir, 'delay_stats_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_delay_distribution(self, data, name, output_dir, color_idx=0):
        """绘制时延分布图"""
        self._ensure_chinese_font()  # 确保中文字体设置
        
        # 过滤有效数据
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            print(f"警告: {name} 没有有效数据")
            return
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 1. 分布直方图
        ax1 = axes[0]
        colors = plt.cm.Set2(color_idx)
        
        # 计算合适的bins数量
        n_bins = min(50, max(10, len(valid_data) // 100))
        
        n, bins, patches = ax1.hist(valid_data, bins=n_bins, alpha=0.7, 
                                   color=colors, edgecolor='black', linewidth=0.5)
        
        # 添加统计信息
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)
        median_val = np.median(valid_data)
        
        ax1.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_val:.2f}')
        ax1.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'中位数: {median_val:.2f}')
        
        ax1.set_xlabel('时延值 (μs)')
        ax1.set_ylabel('频次')
        ax1.set_title(f'{name}分布直方图')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 添加统计文本
        stats_text = (f'样本数: {len(valid_data)}\n'
                     f'均值: {mean_val:.3f} μs\n'
                     f'标准差: {std_val:.3f} μs\n'
                     f'中位数: {median_val:.3f} μs')
        ax1.text(0.65, 0.95, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="wheat", alpha=0.7))
        
        # 2. ECDF图
        ax2 = axes[1]
        sorted_data = np.sort(valid_data)
        ecdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        ax2.plot(sorted_data, ecdf, color=colors, linewidth=2)
        ax2.set_xlabel('时延值 (μs)')
        ax2.set_ylabel('累积概率')
        ax2.set_title(f'{name}经验累积分布函数 (ECDF)')
        ax2.grid(True, alpha=0.3)
        
        # 添加百分位数线
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = np.percentile(valid_data, p)
            ax2.axvline(val, color='gray', linestyle=':', alpha=0.7)
            ax2.text(val, 0.02 + (p/100) * 0.1, f'P{p}', rotation=90, 
                    verticalalignment='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # 保存图片
        safe_name = name.replace('/', '_').replace('\\', '_')
        output_file = os.path.join(output_dir, f'{safe_name}_distribution.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 额外保存ECDF图
        plt.figure(figsize=(8, 6))
        plt.plot(sorted_data, ecdf, color=colors, linewidth=2)
        plt.xlabel('时延值 (μs)')
        plt.ylabel('累积概率')
        plt.title(f'{name}经验累积分布函数 (ECDF)')
        plt.grid(True, alpha=0.3)
        
        # 添加关键百分位数标注
        key_percentiles = [50, 90, 95, 99]
        for p in key_percentiles:
            val = np.percentile(valid_data, p)
            plt.axvline(val, color='red', linestyle='--', alpha=0.7)
            plt.text(val, p/100, f'P{p}: {val:.2f}', rotation=0,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        
        ecdf_file = os.path.join(output_dir, f'{safe_name}_ecdf.png')
        plt.savefig(ecdf_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_gmm_fit(self, data, gmm_params, name, output_dir):
        """绘制GMM拟合结果"""
        self._ensure_chinese_font()  # 确保中文字体设置
        
        # 过滤有效数据
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            print(f"警告: {name} 没有有效数据进行GMM拟合")
            return
        
        # 检查GMM参数
        if not gmm_params or 'weights' not in gmm_params:
            print(f"警告: {name} GMM参数缺失")
            return
        
        # 数据预处理（与GMM拟合时保持一致）
        if gmm_params.get('log_transformed', True):
            # 确保数据为正值
            positive_data = valid_data[valid_data > 0]
            if len(positive_data) == 0:
                print(f"警告: {name} 没有正值数据用于对数变换")
                return
            plot_data = np.log(positive_data)
            data_label = f'{name} (对数变换)'
            x_label = 'log(时延值)'
        else:
            plot_data = valid_data
            data_label = name
            x_label = '时延值 (μs)'
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 绘制原始数据直方图
        n_bins = min(50, max(10, len(plot_data) // 100))
        n, bins, patches = plt.hist(plot_data, bins=n_bins, density=True, 
                                   alpha=0.6, color='lightblue', 
                                   edgecolor='black', linewidth=0.5,
                                   label='实际数据')
        
        # 准备拟合曲线数据
        x_range = np.linspace(plot_data.min(), plot_data.max(), 1000)
        
        # 提取GMM参数
        weights = np.array(gmm_params['weights'])
        means = np.array(gmm_params['means'])
        stds = np.array(gmm_params['stds'])
        n_components = gmm_params['n_components']
        
        # 计算总的GMM概率密度
        total_pdf = np.zeros_like(x_range)
        
        # 绘制每个组件
        colors = plt.cm.tab10(np.linspace(0, 1, n_components))
        
        for i in range(n_components):
            # 单个高斯组件
            component_pdf = weights[i] * norm.pdf(x_range, means[i], stds[i])
            total_pdf += component_pdf
            
            # 绘制单个组件
            plt.plot(x_range, component_pdf, '--', color=colors[i], 
                    linewidth=1.5, alpha=0.8,
                    label=f'组件{i+1} (权重={weights[i]:.3f})')
        
        # 绘制总的拟合曲线
        plt.plot(x_range, total_pdf, 'r-', linewidth=3, alpha=0.9,
                label=f'GMM拟合 ({n_components}组件)')
        
        plt.xlabel(x_label)
        plt.ylabel('概率密度')
        plt.title(f'{data_label} - GMM拟合结果')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # 添加拟合质量信息
        fit_info = (f'组件数: {n_components}\n'
                   f'BIC: {gmm_params.get("bic", "N/A"):.2f}\n'
                   f'AIC: {gmm_params.get("aic", "N/A"):.2f}\n'
                   f'样本数: {len(plot_data):,}')
        
        plt.text(0.02, 0.98, fit_info, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                facecolor="wheat", alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        safe_name = name.replace('/', '_').replace('\\', '_')
        output_file = os.path.join(output_dir, f'{safe_name}_gmm_fit.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"GMM拟合图已保存: {output_file}")
