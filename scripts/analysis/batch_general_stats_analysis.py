import os
import pandas as pd
import subprocess
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.analysis.general_stats_analysis import general_stats_analysis

def batch_analyze(log_files, output_dir, name_map=None, show_plots=False):
    os.makedirs(output_dir, exist_ok=True)
    summary = []
    for log_path in log_files:
        name = os.path.splitext(os.path.basename(log_path))[0]
        if name_map and name in name_map:
            display_name = name_map[name]
        else:
            display_name = name
        print(f"分析: {log_path} -> {display_name}")
        
        # 处理输出目录，确保是有效的目录名
        safe_name = ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in display_name)
        out_dir = os.path.join(output_dir, safe_name)
        
        # 确保输出目录和所有子目录都存在
        os.makedirs(out_dir, exist_ok=True)
        
        # 创建可能需要的子目录
        subdirs = ['plots', 'stats', 'fit']
        for subdir in subdirs:
            os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)
            
        print(f"输出目录: {out_dir}")
        
        try:
            # 调用分析函数，控制是否生成图片
            general_stats_analysis(
                log_path,
                output_dir=out_dir,
                plot_real_time=False,
                filter_percent=0.05,
                piecewise_fit=True,
                piece_num=10,
                plot_raw_delay=show_plots,
                show_plots=show_plots
            )
        except Exception as e:
            print(f"错误: 处理 {log_path} 时发生异常: {e}")
            continue
        
        # 读取统计量
        stats_path = os.path.join(out_dir, 'delay_stats_summary.csv')
        if os.path.exists(stats_path):
            try:
                stats = pd.read_csv(stats_path, index_col=0)
                stats['log_name'] = display_name
                summary.append(stats)
            except Exception as e:
                print(f"警告: 无法读取 {stats_path}: {e}")
    
    # 合并所有统计量
    if summary:
        try:
            all_stats = pd.concat(summary, keys=[s['log_name'][0] for s in summary])
            all_stats_path = os.path.join(output_dir, 'all_delay_stats_summary.csv')
            all_stats.to_csv(all_stats_path)
            print(f"已保存所有统计量到: {all_stats_path}")
        except Exception as e:
            print(f"错误: 合并统计量失败: {e}")
    else:
        print("未找到任何统计量文件！")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_plots', action='store_true', help='是否显示所有分析图')
    args = parser.parse_args()
    # CPU负载日志
    cpu_dir = 'data/load_tests_0612/cpu_load/CPU负载'
    cpu_logs = [os.path.join(cpu_dir, f) for f in os.listdir(cpu_dir) if f.startswith('log') and f.endswith('.log')]
    # 流量负载日志
    traffic_dir = 'data/load_tests_0612/traffic_load/流量负载'
    traffic_logs = [os.path.join(traffic_dir, f) for f in os.listdir(traffic_dir) if f.startswith('log') and f.endswith('.log')]
    all_logs = cpu_logs + traffic_logs
    # 可选：自定义名称映射
    name_map = None
    batch_analyze(all_logs, output_dir='results/batch_stats', name_map=name_map, show_plots=args.show_plots)
