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
        if df.shape[1] < 4:
            df = pd.read_csv(log_file, sep=' ', header=None)
        
        # 确保至少有4个时间戳
        if df.shape[1] < 4:
            raise ValueError(f"日志文件格式不正确，只有{df.shape[1]}列")
        
        # 提取t1, t2, t3, t4
        df['t1'] = df.iloc[:, 0]
        df['t2'] = df.iloc[:, 1] 
        df['t3'] = df.iloc[:, 2]
        df['t4'] = df.iloc[:, 3]
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

def calculate_statistics(df):
    """计算平均时延统计量"""
    stats_dict = {}
    
    data = df['avg_delay'].dropna()
    if len(data) > 0:
        stats_dict['avg_delay'] = {
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
    
    return stats_dict

def create_comparison_plots(original_df, processed_df, output_dir):
    """创建平均时延分布密度直方图对比和CDF对比"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建2x2子图布局
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('平均时延分布对比分析', fontsize=16, fontweight='bold')
    
    # 1. 密度直方图对比（左上）
    ax1.hist(original_df['avg_delay'], bins=50, alpha=0.7, label='原始日志', 
             density=True, color='blue', edgecolor='black', linewidth=0.5)
    ax1.hist(processed_df['avg_delay'], bins=50, alpha=0.7, label='处理后日志', 
             density=True, color='red', edgecolor='black', linewidth=0.5)
    
    ax1.set_xlabel('平均时延 (μs)')
    ax1.set_ylabel('密度')
    ax1.set_title('平均时延分布密度直方图对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 添加统计信息
    orig_mean = original_df['avg_delay'].mean()
    orig_std = original_df['avg_delay'].std()
    proc_mean = processed_df['avg_delay'].mean()
    proc_std = processed_df['avg_delay'].std()
    
    textstr = f'原始: μ={orig_mean:.2f}μs, σ={orig_std:.2f}μs\n处理后: μ={proc_mean:.2f}μs, σ={proc_std:.2f}μs'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. CDF对比图（右上）
    # 计算CDF
    orig_data = np.sort(original_df['avg_delay'].dropna())
    proc_data = np.sort(processed_df['avg_delay'].dropna())
    
    orig_cdf_y = np.arange(1, len(orig_data) + 1) / len(orig_data)
    proc_cdf_y = np.arange(1, len(proc_data) + 1) / len(proc_data)
    
    ax2.plot(orig_data, orig_cdf_y, label='原始日志', color='blue', linewidth=2)
    ax2.plot(proc_data, proc_cdf_y, label='处理后日志', color='red', linewidth=2)
    
    ax2.set_xlabel('平均时延 (μs)')
    ax2.set_ylabel('累积概率')
    ax2.set_title('平均时延累积分布函数(CDF)对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 添加关键分位数标记
    for percentile in [50, 90, 95, 99]:
        orig_val = np.percentile(orig_data, percentile)
        proc_val = np.percentile(proc_data, percentile)
        ax2.axhline(y=percentile/100, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=orig_val, color='blue', linestyle=':', alpha=0.7)
        ax2.axvline(x=proc_val, color='red', linestyle=':', alpha=0.7)
    
    # 3. 原始数据单独分布（左下）
    ax3.hist(original_df['avg_delay'], bins=50, alpha=0.8, color='blue', 
             density=True, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('平均时延 (μs)')
    ax3.set_ylabel('密度')
    ax3.set_title(f'原始数据分布 (n={len(original_df)})')
    ax3.grid(True, alpha=0.3)
    
    # 4. 处理后数据单独分布（右下）
    ax4.hist(processed_df['avg_delay'], bins=50, alpha=0.8, color='red', 
             density=True, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('平均时延 (μs)')
    ax4.set_ylabel('密度')
    ax4.set_title(f'处理后数据分布 (n={len(processed_df)})')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "avg_delay_distribution_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "avg_delay_distribution_comparison.svg", bbox_inches='tight')
    plt.close()
    
    # 单独创建高分辨率CDF对比图
    plt.figure(figsize=(10, 8))
    plt.plot(orig_data, orig_cdf_y, label='原始日志', color='blue', linewidth=2)
    plt.plot(proc_data, proc_cdf_y, label='处理后日志', color='red', linewidth=2)
    
    plt.xlabel('平均时延 (μs)')
    plt.ylabel('累积概率')
    plt.title('平均时延累积分布函数(CDF)对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加关键分位数标记和标签
    percentiles = [50, 90, 95, 99]
    colors = ['gray', 'orange', 'purple', 'brown']
    
    for i, percentile in enumerate(percentiles):
        orig_val = np.percentile(orig_data, percentile)
        proc_val = np.percentile(proc_data, percentile)
        y_val = percentile/100
        
        plt.axhline(y=y_val, color=colors[i], linestyle='--', alpha=0.6, 
                   label=f'P{percentile}')
        plt.axvline(x=orig_val, color='blue', linestyle=':', alpha=0.5)
        plt.axvline(x=proc_val, color='red', linestyle=':', alpha=0.5)
        
        # 添加数值标签
        plt.text(orig_val, y_val + 0.02, f'{orig_val:.1f}', 
                color='blue', fontsize=8, ha='center')
        plt.text(proc_val, y_val - 0.02, f'{proc_val:.1f}', 
                color='red', fontsize=8, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / "avg_delay_cdf_comparison.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "avg_delay_cdf_comparison.svg", bbox_inches='tight')
    plt.close()
    
    print(f"平均时延分布对比图已保存到: {output_dir}")
    print(f"CDF最大差值: {ks_stat:.4f} (位置: {max_diff_x:.2f}μs)")
    print(f"对数尺度KS统计量: {ks_stat_log:.4f}")
    
    return ks_stat, max_diff_x

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
    
    # 保存为CSV
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / "statistics_comparison.csv", index=False)
    
    print(f"统计量对比已保存到: {output_dir}")
    return comparison_df

def generate_report(original_stats, processed_stats, comparison_df, output_dir):
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
    
    # 统计量对比
    report.append("## 主要统计量对比")
    report.append("| 统计量 | 原始日志 | 处理后日志 | 差异 | 相对变化(%) |")
    report.append("|--------|----------|------------|------|-------------|")
    
    for _, row in comparison_df.iterrows():
        report.append(f"| {row['metric']} | "
                     f"{row['original']:.4f} | {row['processed']:.4f} | "
                     f"{row['difference']:.4f} | {row['relative_change_pct']:.2f}% |")
    
    report.append("")
    
    # 保存报告
    with open(output_dir / "analysis_report.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"分析报告已保存到: {output_dir / 'analysis_report.md'}")

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
        return
    
    # 计算统计量
    print("计算统计量...")
    original_stats = calculate_statistics(original_df)
    processed_stats = calculate_statistics(processed_df)
    
    # 保存统计量
    with open(output_dir / "original_stats.json", 'w') as f:
        json.dump(original_stats, f, indent=2)
    with open(output_dir / "processed_stats.json", 'w') as f:
        json.dump(processed_stats, f, indent=2)
    
    # 创建对比图表
    print("生成平均时延分布对比图...")
    create_comparison_plots(original_df, processed_df, output_dir)
    
    # 创建统计量对比
    comparison_df = create_statistics_comparison(original_stats, processed_stats, output_dir)
    
    # 生成报告
    print("生成分析报告...")
    generate_report(original_stats, processed_stats, comparison_df, output_dir)
    
    print(f"\n对比分析完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    # 解决中文字体问题
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    main()
