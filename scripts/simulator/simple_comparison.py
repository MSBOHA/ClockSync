#!/usr/bin/env python3
"""
简化版时钟同步日志对比分析脚本
只输出平均时延分布密度直方图对比
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import argparse
from scipy import stats
import json

def load_sync_log(log_file):
    """
    加载时钟同步日志文件
    
    Args:
        log_file: 日志文件路径
        
    Returns:
        DataFrame: 包含时延数据的DataFrame
    """
    try:
        # 尝试读取日志文件，假设是制表符分隔
        df = pd.read_csv(log_file, sep='\t', header=None)
          # 如果列数不对，尝试其他分隔符
        if df.shape[1] < 14:  # 需要至少14列才能访问RAW数据
            df = pd.read_csv(log_file, sep=' ', header=None)
        
        # 确保至少有14个列以访问RAW时间戳
        if df.shape[1] < 14:
            raise ValueError(f"日志文件格式不正确，只有{df.shape[1]}列，需要至少14列来访问RAW数据")
        
        # 提取RAW数据的t1, t2, t3, t4 (第11-14列，索引10-13)
        df['t1'] = df.iloc[:, 10]  # 第11列
        df['t2'] = df.iloc[:, 11]  # 第12列
        df['t3'] = df.iloc[:, 12]  # 第13列
        df['t4'] = df.iloc[:, 13]  # 第14列
          # 使用NTP标准公式计算平均时延（网络往返时延的一半）
        # 平均时延 = ((t4-t1) - (t3-t2)) / 2，单位转换为微秒
        df['avg_delay'] = (((df['t4'] - df['t1']) - (df['t3'] - df['t2'])) / 2) * 10**6  # ns to us
        
        print(f"成功加载日志: {log_file}")
        print(f"数据行数: {len(df)}")
        print(f"平均时延范围: {df['avg_delay'].min():.2f} ~ {df['avg_delay'].max():.2f} us")
        
        return df
        
    except Exception as e:
        print(f"加载日志文件失败: {e}")
        return None

def calculate_statistics(df, apply_filter=True, lower_percentile=5, upper_percentile=95):
    """
    计算平均时延统计量
    
    Args:
        df: 数据框
        apply_filter: 是否应用分位筛选
        lower_percentile: 下分位数阈值
        upper_percentile: 上分位数阈值
    """
    stats_dict = {}
    
    data = df['avg_delay'].dropna()
    if len(data) > 0:
        # 原始数据统计
        stats_dict['avg_delay_raw'] = {
            'count': len(data),
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'q25': data.quantile(0.25),
            'q50': data.quantile(0.50),
            'q75': data.quantile(0.75),
            'q90': data.quantile(0.90),
            'q95': data.quantile(0.95),
            'q99': data.quantile(0.99),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
        
        # 筛选后数据统计
        if apply_filter:
            filtered_data, filter_stats = filter_outliers(data, lower_percentile, upper_percentile)
            stats_dict['filter_info'] = filter_stats
            
            if len(filtered_data) > 0:
                stats_dict['avg_delay'] = {
                    'count': len(filtered_data),
                    'mean': filtered_data.mean(),
                    'std': filtered_data.std(),
                    'min': filtered_data.min(),
                    'max': filtered_data.max(),
                    'q25': np.percentile(filtered_data, 25),
                    'q50': np.percentile(filtered_data, 50),
                    'q75': np.percentile(filtered_data, 75),
                    'q90': np.percentile(filtered_data, 90),
                    'q95': np.percentile(filtered_data, 95),
                    'q99': np.percentile(filtered_data, 99),
                    'skewness': stats.skew(filtered_data),
                    'kurtosis': stats.kurtosis(filtered_data)
                }
                # 保存筛选后的数据供后续使用
                stats_dict['filtered_data'] = filtered_data
            else:
                stats_dict['avg_delay'] = stats_dict['avg_delay_raw']
                stats_dict['filtered_data'] = data
        else:
            stats_dict['avg_delay'] = stats_dict['avg_delay_raw']
            stats_dict['filtered_data'] = data
    
    return stats_dict

def create_comparison_plots(original_stats, processed_stats, output_dir):
    """
    创建平均时延分布密度直方图和CDF对比
    
    Args:
        original_stats: 原始数据统计信息（包含筛选后数据）
        processed_stats: 处理后数据统计信息（包含筛选后数据）
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取筛选后的数据
    original_data = original_stats['filtered_data']
    processed_data = processed_stats['filtered_data']
    
    # 创建包含两个子图的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1: 平均时延分布密度直方图对比（使用筛选后数据）
    ax1.hist(original_data, bins=50, alpha=0.7, label='原始日志(筛选后)', 
             density=True, color='blue', edgecolor='black', linewidth=0.5)
    ax1.hist(processed_data, bins=50, alpha=0.7, label='处理后日志(筛选后)', 
             density=True, color='red', edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('平均时延 (μs)')
    ax1.set_ylabel('密度')
    ax1.set_title('平均时延分布密度直方图对比（筛选后）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息到密度图
    orig_mean = original_stats['avg_delay']['mean']
    orig_std = original_stats['avg_delay']['std']
    proc_mean = processed_stats['avg_delay']['mean']
    proc_std = processed_stats['avg_delay']['std']
    
    textstr = f'原始日志: μ={orig_mean:.2f}μs, σ={orig_std:.2f}μs\n处理后日志: μ={proc_mean:.2f}μs, σ={proc_std:.2f}μs'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 显示筛选信息
    if 'filter_info' in original_stats:
        filter_text = f"筛选: {original_stats['filter_info']['lower_percentile']}-{original_stats['filter_info']['upper_percentile']}%分位\n"
        filter_text += f"原始: {original_stats['filter_info']['original_count']} -> 筛选后: {original_stats['filter_info']['filtered_count']}"
        ax1.text(0.98, 0.98, filter_text, transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))    
    # 子图2: CDF对比（使用筛选后数据）
    # 获取数据并排序
    orig_data = np.sort(original_data)
    proc_data = np.sort(processed_data)
    
    print(f"原始数据样本数（筛选后）: {len(orig_data)}")
    print(f"处理后数据样本数（筛选后）: {len(proc_data)}")
    print(f"原始数据范围: {orig_data.min():.2f} ~ {orig_data.max():.2f} μs")
    print(f"处理后数据范围: {proc_data.min():.2f} ~ {proc_data.max():.2f} μs")
    
    # 计算累积概率
    orig_cdf = np.arange(1, len(orig_data) + 1) / len(orig_data)
    proc_cdf = np.arange(1, len(proc_data) + 1) / len(proc_data)
      # 使用scipy.stats中的更准确方法计算KS统计量
    from scipy.stats import ks_2samp, entropy
    ks_stat, p_value = ks_2samp(orig_data, proc_data)
    
    # 计算KL散度（Kullback-Leibler divergence）
    # 首先需要将数据转换为概率分布
    def compute_kl_divergence(data1, data2, bins=50):
        """计算两个数据集之间的KL散度"""
        # 确定共同的范围和bins
        min_val = min(data1.min(), data2.min())
        max_val = max(data1.max(), data2.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # 计算直方图（概率密度）
        hist1, _ = np.histogram(data1, bins=bin_edges, density=True)
        hist2, _ = np.histogram(data2, bins=bin_edges, density=True)
        
        # 归一化为概率分布
        hist1 = hist1 / hist1.sum()
        hist2 = hist2 / hist2.sum()
        
        # 添加小的epsilon避免log(0)
        epsilon = 1e-10
        hist1 = hist1 + epsilon
        hist2 = hist2 + epsilon
        
        # 计算KL散度 D(P||Q) = sum(P * log(P/Q))
        kl_div = entropy(hist1, hist2)
        return kl_div
    
    kl_divergence = compute_kl_divergence(orig_data, proc_data)
    
    # 手动验证：计算CDF最大差值
    # 只在数据重叠的范围内比较
    overlap_min = max(orig_data.min(), proc_data.min())
    overlap_max = min(orig_data.max(), proc_data.max())
    
    # 在重叠范围内创建更密集的采样点
    x_overlap = np.linspace(overlap_min, overlap_max, 2000)
    
    # 插值计算在相同x值上的CDF
    orig_cdf_interp = np.interp(x_overlap, orig_data, orig_cdf)
    proc_cdf_interp = np.interp(x_overlap, proc_data, proc_cdf)
    
    # 计算差值
    cdf_diff = np.abs(orig_cdf_interp - proc_cdf_interp)
    max_diff_manual = np.max(cdf_diff)
    max_diff_idx = np.argmax(cdf_diff)
    max_diff_x = x_overlap[max_diff_idx]
    
    print(f"scipy KS统计量: {ks_stat:.6f}")
    print(f"手动计算最大CDF差值: {max_diff_manual:.6f}")
    print(f"最大差值位置: {max_diff_x:.2f} μs")
    
    # 在最大差值点附近检查详细情况
    window_size = 10  # 检查±10μs范围
    window_mask = (x_overlap >= max_diff_x - window_size) & (x_overlap <= max_diff_x + window_size)
    window_x = x_overlap[window_mask]
    window_orig = orig_cdf_interp[window_mask]
    window_proc = proc_cdf_interp[window_mask]
    window_diff = cdf_diff[window_mask]
    
    print(f"最大差值点附近的CDF值:")
    print(f"  位置: {max_diff_x:.2f} μs")
    print(f"  原始CDF: {orig_cdf_interp[max_diff_idx]:.6f}")
    print(f"  处理后CDF: {proc_cdf_interp[max_diff_idx]:.6f}")
    print(f"  差值: {window_diff.max():.6f}")
    
    # 绘制CDF，使用scipy的KS统计量
    ax2.plot(orig_data, orig_cdf, label='原始日志', color='blue', linewidth=2)
    ax2.plot(proc_data, proc_cdf, label='处理后日志', color='red', linewidth=2)
    
    # 标记最大差值点
    ax2.axvline(x=max_diff_x, color='green', linestyle='--', alpha=0.7, linewidth=1)
    ax2.text(max_diff_x, 0.5, f'最大CDF差值\n{ks_stat:.4f}\n@{max_diff_x:.2f}μs', 
             ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    ax2.set_xlabel('平均时延 (μs)')
    ax2.set_ylabel('累积概率')
    ax2.set_title('平均时延累积分布函数(CDF)对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
      # 添加CDF统计信息，移除p值，添加KL散度
    textstr_cdf = f'KS统计量: {ks_stat:.4f}\nKL散度: {kl_divergence:.4f}\n最大差值位置: {max_diff_x:.2f}μs'
    ax2.text(0.02, 0.98, textstr_cdf, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "avg_delay_distribution_and_cdf_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 单独保存CDF图
    plt.figure(figsize=(10, 6))
    plt.plot(orig_data, orig_cdf, label='原始日志', color='blue', linewidth=2)
    plt.plot(proc_data, proc_cdf, label='处理后日志', color='red', linewidth=2)
    
    # 标记最大差值点
    plt.axvline(x=max_diff_x, color='green', linestyle='--', alpha=0.7, linewidth=1)
    plt.text(max_diff_x, 0.5, f'最大CDF差值: {ks_stat:.4f}\n位置: {max_diff_x:.2f}μs', 
             ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.xlabel('平均时延 (μs)')
    plt.ylabel('累积概率')
    plt.title('平均时延累积分布函数(CDF)对比')
    plt.legend()
    plt.grid(True, alpha=0.3)    # 添加详细统计信息，移除p值，添加KL散度
    textstr_detailed = (f'原始日志: 中位数={np.median(orig_data):.2f}μs\n'
                       f'处理后日志: 中位数={np.median(proc_data):.2f}μs\n'
                       f'KS统计量: {ks_stat:.4f}\n'
                       f'KL散度: {kl_divergence:.4f}')
    plt.text(0.02, 0.98, textstr_detailed, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "avg_delay_cdf_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()    # 创建对数尺度的密度直方图和CDF对比
    fig_log, (ax_log1, ax_log2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子图1: 对数尺度密度直方图对比
    # 过滤掉非正值以避免对数变换问题
    orig_data_pos = original_data[original_data > 0]
    proc_data_pos = processed_data[processed_data > 0]
    
    ax_log1.hist(np.log(orig_data_pos), bins=50, alpha=0.7, label='原始日志(ln,筛选后)', 
                 density=True, color='blue', edgecolor='black', linewidth=0.5)
    ax_log1.hist(np.log(proc_data_pos), bins=50, alpha=0.7, label='处理后日志(ln,筛选后)', 
                 density=True, color='red', edgecolor='black', linewidth=0.5)
    
    ax_log1.set_xlabel('平均时延 (ln μs)')
    ax_log1.set_ylabel('密度')
    ax_log1.set_title('平均时延分布密度直方图对比 (自然对数尺度，筛选后)')
    ax_log1.legend()
    ax_log1.grid(True, alpha=0.3)
    
    # 添加对数尺度统计信息
    orig_log_mean = np.mean(np.log(orig_data_pos))
    orig_log_std = np.std(np.log(orig_data_pos))
    proc_log_mean = np.mean(np.log(proc_data_pos))
    proc_log_std = np.std(np.log(proc_data_pos))
    
    textstr_log = (f'原始日志(ln): μ={orig_log_mean:.3f}, σ={orig_log_std:.3f}\n'
                   f'处理后日志(ln): μ={proc_log_mean:.3f}, σ={proc_log_std:.3f}')
    ax_log1.text(0.02, 0.98, textstr_log, transform=ax_log1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 子图2: 对数尺度CDF对比
    orig_data_log = np.sort(np.log(orig_data_pos))
    proc_data_log = np.sort(np.log(proc_data_pos))
    
    orig_cdf_log = np.arange(1, len(orig_data_log) + 1) / len(orig_data_log)
    proc_cdf_log = np.arange(1, len(proc_data_log) + 1) / len(proc_data_log)
      # 计算对数尺度的KS统计量和KL散度
    ks_stat_log, p_value_log = ks_2samp(orig_data_log, proc_data_log)
    kl_divergence_log = compute_kl_divergence(orig_data_log, proc_data_log)
    ax_log2.plot(orig_data_log, orig_cdf_log, label='原始日志(ln)', color='blue', linewidth=2)
    ax_log2.plot(proc_data_log, proc_cdf_log, label='处理后日志(ln)', color='red', linewidth=2)
    
    ax_log2.set_xlabel('平均时延 (ln μs)')
    ax_log2.set_ylabel('累积概率')
    ax_log2.set_title('平均时延累积分布函数(CDF)对比 (自然对数尺度)')
    ax_log2.legend()
    ax_log2.grid(True, alpha=0.3)
    
    # 添加对数尺度CDF统计信息，移除p值，添加KL散度
    textstr_cdf_log = (f'自然对数尺度KS统计量: {ks_stat_log:.4f}\n'
                       f'自然对数尺度KL散度: {kl_divergence_log:.4f}\n'
                       f'中位数差异: {np.median(proc_data_log) - np.median(orig_data_log):.3f}')
    ax_log2.text(0.02, 0.98, textstr_cdf_log, transform=ax_log2.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "avg_delay_distribution_and_cdf_comparison_log_scale.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 单独保存对数尺度CDF图
    plt.figure(figsize=(10, 6))
    plt.plot(orig_data_log, orig_cdf_log, label='原始日志(ln)', color='blue', linewidth=2)
    plt.plot(proc_data_log, proc_cdf_log, label='处理后日志(ln)', color='red', linewidth=2)
    
    plt.xlabel('平均时延 (ln μs)')
    plt.ylabel('累积概率')
    plt.title('平均时延累积分布函数(CDF)对比 (自然对数尺度)')
    plt.legend()
    plt.grid(True, alpha=0.3)    # 添加详细的对数尺度统计信息，移除p值，添加KL散度
    textstr_detailed_log = (f'原始日志(ln): 中位数={np.median(orig_data_log):.3f}\n'
                           f'处理后日志(ln): 中位数={np.median(proc_data_log):.3f}\n'
                           f'自然对数尺度KS统计量: {ks_stat_log:.4f}\n'
                           f'自然对数尺度KL散度: {kl_divergence_log:.4f}')
    plt.text(0.02, 0.98, textstr_detailed_log, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / "avg_delay_cdf_comparison_log_scale.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"平均时延分布对比图已保存到: {output_dir}")
    print(f"CDF最大差值: {ks_stat:.4f} (位置: {max_diff_x:.2f}μs)")
    print(f"KL散度: {kl_divergence:.4f}")
    print(f"自然对数尺度KS统计量: {ks_stat_log:.4f}")
    print(f"自然对数尺度KL散度: {kl_divergence_log:.4f}")
    
    return ks_stat, max_diff_x, kl_divergence

def create_statistics_comparison(original_stats, processed_stats, output_dir, max_cdf_diff=None, max_diff_x=None):
    """创建统计量对比表格"""
    output_dir = Path(output_dir)
    
    # 创建对比表格
    comparison_data = []
    
    if 'avg_delay' in original_stats and 'avg_delay' in processed_stats:
        orig = original_stats['avg_delay']
        proc = processed_stats['avg_delay']
        
        for metric in ['mean', 'std', 'min', 'max', 'q50', 'q95']:
            if metric in orig and metric in proc:
                diff = proc[metric] - orig[metric]
                rel_change = (diff / orig[metric] * 100) if orig[metric] != 0 else np.nan
                
                comparison_data.append({
                    'delay_type': 'avg_delay',
                    'metric': metric,
                    'original': orig[metric],
                    'processed': proc[metric],
                    'difference': diff,
                    'relative_change_pct': rel_change
                })
        
        # 添加CDF最大差值信息
        if max_cdf_diff is not None:
            comparison_data.append({
                'delay_type': 'avg_delay',
                'metric': 'max_cdf_diff',
                'original': 0.0,
                'processed': max_cdf_diff,
                'difference': max_cdf_diff,
                'relative_change_pct': np.inf
            })
            
        if max_diff_x is not None:
            comparison_data.append({
                'delay_type': 'avg_delay',
                'metric': 'max_diff_position_us',
                'original': 0.0,
                'processed': max_diff_x,
                'difference': max_diff_x,
                'relative_change_pct': np.inf
            })
    
    # 保存为CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / "statistics_comparison.csv", index=False)
    
    print(f"统计量对比已保存到: {output_dir}")
    return comparison_df

def generate_report(original_stats, processed_stats, comparison_df, output_dir, max_cdf_diff=None, max_diff_x=None):
    """生成对比分析报告"""
    
    report = []
    report.append("# 时钟同步日志平均时延对比分析报告")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # 数据概览
    report.append("## 数据概览")
    if 'avg_delay' in original_stats and 'avg_delay' in processed_stats:
        orig = original_stats['avg_delay']
        proc = processed_stats['avg_delay']
        
        report.append(f"### 平均时延统计")
        report.append(f"- 原始日志数据点数: {orig['count']}")
        report.append(f"- 处理后日志数据点数: {proc['count']}")
        report.append("")
    
    # CDF对比分析
    if max_cdf_diff is not None:
        report.append("## CDF对比分析")
        report.append("### Kolmogorov-Smirnov统计量")
        report.append(f"- 最大CDF差值: {max_cdf_diff:.6f}")
        if max_diff_x is not None:
            report.append(f"- 最大差值位置: {max_diff_x:.4f} μs")
        report.append("")
        
        # 解释KS统计量的含义
        if max_cdf_diff < 0.2:
            significance = "较小，两个分布相似"
        elif max_cdf_diff < 0.4:
            significance = "中等，两个分布有一定差异"
        else:
            significance = "较大，表明两个分布存在显著差异"
        
        report.append(f"- **统计意义**: KS统计量为{max_cdf_diff:.4f}，{significance}")
        report.append("")
    
    # 统计量对比
    report.append("## 主要统计量对比")
    report.append("| 统计量 | 原始日志 | 处理后日志 | 差异 | 相对变化(%) |")
    report.append("|--------|----------|------------|------|-------------|")
    
    for _, row in comparison_df.iterrows():
        if row['metric'] not in ['max_cdf_diff', 'max_diff_position_us']:  # 只显示基本统计量
            report.append(f"| {row['metric']} | "
                         f"{row['original']:.4f} | {row['processed']:.4f} | "
                         f"{row['difference']:.4f} | {row['relative_change_pct']:.2f}% |")
    
    report.append("")
    
    # CDF分析结果单独列出
    if max_cdf_diff is not None:
        report.append("## CDF分析结果")
        report.append("| 指标 | 数值 |")
        report.append("|------|------|")
        report.append(f"| 最大CDF差值 | {max_cdf_diff:.6f} |")
        if max_diff_x is not None:
            report.append(f"| 最大差值位置(μs) | {max_diff_x:.4f} |")
        report.append("")
    
    # 保存报告
    with open(output_dir / "analysis_report.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"分析报告已保存到: {output_dir / 'analysis_report.md'}")

def filter_outliers(data, lower_percentile=5, upper_percentile=95):
    """
    基于分位数筛选数据，去除异常值
    
    Args:
        data: 输入数据数组
        lower_percentile: 下分位数阈值 (默认5%)
        upper_percentile: 上分位数阈值 (默认95%)
        
    Returns:
        筛选后的数据和筛选统计信息
    """
    if len(data) == 0:
        return data, {"filtered_count": 0, "original_count": 0}
    
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    
    # 筛选数据
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    filter_stats = {
        "original_count": len(data),
        "filtered_count": len(filtered_data),
        "removed_count": len(data) - len(filtered_data),
        "removal_rate": (len(data) - len(filtered_data)) / len(data) * 100,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "lower_percentile": lower_percentile,
        "upper_percentile": upper_percentile
    }
    
    print(f"分位筛选: {filter_stats['original_count']} -> {filter_stats['filtered_count']} "
          f"(移除 {filter_stats['removed_count']} 个异常值, {filter_stats['removal_rate']:.1f}%)")
    print(f"筛选范围: [{lower_bound:.2f}, {upper_bound:.2f}] μs")
    
    return filtered_data, filter_stats

def main():
    parser = argparse.ArgumentParser(description='时钟同步日志对比分析')
    parser.add_argument('--original_log', required=True, help='原始日志文件路径')
    parser.add_argument('--processed_log', required=True, help='处理后日志文件路径')
    parser.add_argument('--output_dir', default='simple_analysis_results', help='输出目录')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.original_log):
        print(f"错误: 原始日志文件不存在: {args.original_log}")
        return
    
    if not os.path.exists(args.processed_log):
        print(f"错误: 处理后日志文件不存在: {args.processed_log}")
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("开始加载和分析日志文件...")
    
    # 加载日志数据
    original_df = load_sync_log(args.original_log)
    if original_df is None:
        print("加载原始日志失败")
        return
    
    processed_df = load_sync_log(args.processed_log)
    if processed_df is None:
        print("加载处理后日志失败")
        return    # 计算统计量（原始数据筛选，处理后数据不筛选）
    print("计算统计量（原始数据应用1%-99%分位筛选，处理后数据不筛选）...")
    original_stats = calculate_statistics(original_df, apply_filter=True, lower_percentile=1, upper_percentile=99)
    processed_stats = calculate_statistics(processed_df, apply_filter=False)

    # 保存统计量
    with open(output_dir / "original_stats.json", 'w') as f:
        # 转换numpy数组为列表以便JSON序列化
        stats_for_json = {}
        for key, value in original_stats.items():
            if key == 'filtered_data':
                continue  # 不保存数据数组
            stats_for_json[key] = value
        json.dump(stats_for_json, f, indent=2)
    
    with open(output_dir / "processed_stats.json", 'w') as f:
        # 转换numpy数组为列表以便JSON序列化
        stats_for_json = {}
        for key, value in processed_stats.items():
            if key == 'filtered_data':
                continue  # 不保存数据数组
            stats_for_json[key] = value
        json.dump(stats_for_json, f, indent=2)
      # 创建对比图表（使用筛选后的数据）
    print("生成平均时延分布对比图（基于筛选后数据）...")
    ks_stat, max_diff_x, kl_divergence = create_comparison_plots(original_stats, processed_stats, output_dir)
    
    # 创建统计量对比
    comparison_df = create_statistics_comparison(original_stats, processed_stats, output_dir, ks_stat, max_diff_x)
    
    # 生成报告
    print("生成分析报告...")
    generate_report(original_stats, processed_stats, comparison_df, output_dir, ks_stat, max_diff_x)
    
    print(f"\n对比分析完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    # 解决中文字体问题
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    main()
