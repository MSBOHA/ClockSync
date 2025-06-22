"""
重构后的时钟同步分析主模块
整合所有功能模块，提供简洁的分析接口
"""
import os
import sys
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor, OffsetFitter, GMMFitter
from stats_calculator import StatisticsCalculator
from visualizer import Visualizer
from config import AnalysisConfig
from file_manager import FileManager


class ClockSyncAnalyzer:
    """时钟同步分析器主类"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.config.validate()
        
        # 初始化各个模块
        self.data_processor = DataProcessor()
        self.offset_fitter = OffsetFitter(method=config.fit_method)
        self.gmm_fitter = GMMFitter(
            n_components_range=config.gmm_n_components_range,
            log_transform=config.gmm_log_transform
        ) if config.enable_gmm else None
        self.stats_calculator = StatisticsCalculator()
        self.visualizer = Visualizer()
        self.file_manager = FileManager(config.output_dir)
        
        print(f"分析器初始化完成，输出目录: {config.output_dir}")
    
    def analyze_single_file(self, log_file):
        """分析单个日志文件"""
        print(f"\n开始分析文件: {log_file}")
        print("=" * 60)
        
        try:
            # 1. 数据加载和预处理
            df = self.data_processor.load_data(log_file)
            df_filtered = self.data_processor.filter_outliers(
                df, self.config.lower_percent, self.config.upper_percent
            )
            
            # 提取负载等级
            load_level = self.data_processor.extract_load_level(log_file)
            print(f"识别的负载等级: {load_level}")
            
            # 为特定负载等级创建输出目录
            if load_level:
                load_output_dir = self.file_manager.create_load_level_dir(load_level)
            else:
                load_output_dir = self.config.output_dir
            
            # 2. Offset拟合
            print("\n进行Offset拟合...")
            offset_est = self.offset_fitter.fit(
                df_filtered,
                self.config.fit_lower_percent,
                self.config.fit_upper_percent,
                self.config.piecewise_fit,
                self.config.piece_num
            )
            
            # 计算拟合质量
            if offset_est is not None and not np.isnan(offset_est).all():
                fit_quality = self.stats_calculator.calculate_fit_quality(
                    df_filtered["ns2us(getOffset(RAW))"].values, offset_est
                )
                print(f"拟合质量 - RMSE: {fit_quality.get('rmse', 'N/A'):.4f}, R²: {fit_quality.get('r2', 'N/A'):.4f}")
            else:
                fit_quality = {'rmse': np.nan, 'r2': np.nan, 'mae': np.nan, 'mse': np.nan}
                print("Offset拟合失败")
            
            # 3. 计算自估时延
            if offset_est is not None and not np.isnan(offset_est).all():
                t2t1_raw = df_filtered["ns2us(getT2T1(RAW))"].values
                t3t4_raw = df_filtered["ns2us(getT3T4(RAW))"].values
                up_delay_est = t2t1_raw - offset_est
                down_delay_est = offset_est - t3t4_raw
            else:
                up_delay_est = np.full(len(df_filtered), np.nan)
                down_delay_est = np.full(len(df_filtered), np.nan)
            
            # 4. 计算统计量
            print("\n计算统计量...")
            delay_stats = self.stats_calculator.calculate_delay_stats(up_delay_est, down_delay_est)
            
            # 5. GMM拟合（如果启用）
            gmm_results = {}
            if self.config.enable_gmm and self.gmm_fitter:
                print("\n进行GMM拟合...")
                
                for data, name in [(up_delay_est, '上行时延'), (down_delay_est, '下行时延')]:
                    if not np.isnan(data).all():
                        gmm_params = self.gmm_fitter.fit_gmm(data, load_level)
                        if gmm_params:
                            gmm_results[name] = gmm_params
                            print(f"{name} GMM拟合完成: {gmm_params['n_components']}个组件")
                            
                            # 保存GMM参数
                            self.file_manager.save_gmm_parameters(
                                gmm_params, name.replace('时延', 'delay'), load_level
                            )
                        else:
                            print(f"{name} GMM拟合失败")
            
            # 6. 生成图表
            print("\n生成可视化图表...")
            plots_dir = os.path.join(load_output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # 原始时间序列图
            self.visualizer.plot_raw_time_series(
                df_filtered, offset_est, fit_quality, plots_dir,
                self.config.piecewise_fit, self.config.piece_num, self.config.fit_method
            )
            
            # 统计量对比图
            if delay_stats:
                self.visualizer.plot_delay_comparison(delay_stats, plots_dir)
            
            # 时延分布图
            for i, (data, name) in enumerate([(up_delay_est, '自估上行时延'), (down_delay_est, '自估下行时延')]):
                if not np.isnan(data).all():
                    self.visualizer.plot_delay_distribution(data, name, plots_dir, i)
                    
                    # GMM拟合图
                    if name.replace('自估', '') in gmm_results:
                        self.visualizer.plot_gmm_fit(
                            data, gmm_results[name.replace('自估', '')], name, plots_dir
                        )
            
            # 实时时间序列图（可选）
            if self.config.plot_real_time:
                self.visualizer.plot_real_time_series(df_filtered, plots_dir)
            
            # 7. 保存结果
            print("\n保存分析结果...")
            
            # 保存统计摘要
            if delay_stats:
                self.file_manager.save_stats_summary(delay_stats, load_level)
            
            # 保存拟合质量
            self.file_manager.save_fit_quality(fit_quality, self.config, load_level)
            
            # 保存时延数据
            if self.config.save_detailed_results:
                self.file_manager.save_delay_data(up_delay_est, down_delay_est, load_level)
              # 准备结果摘要
            result_summary = {
                'load_level': load_level,
                'file_path': log_file,
                'data_points': len(df_filtered),
                'fit_quality': fit_quality,
                'delay_stats': delay_stats,
                'gmm_results': gmm_results,  # 保存完整的GMM结果，包含所有参数
                'output_dir': load_output_dir
            }
            
            print(f"文件 {log_file} 分析完成！")
            return result_summary
            
        except Exception as e:
            print(f"分析失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_batch(self, file_list):
        """批量分析多个文件"""
        print(f"\n开始批量分析 {len(file_list)} 个文件")
        print("=" * 60)
        
        results = []
        failed_files = []
        
        for i, log_file in enumerate(file_list, 1):
            print(f"\n[{i}/{len(file_list)}] 处理文件: {os.path.basename(log_file)}")
            
            result = self.analyze_single_file(log_file)
            if result:
                results.append(result)
            else:
                failed_files.append(log_file)
        
        # 生成批量分析报告
        batch_summary = {
            'total_files': len(file_list),
            'successful_files': len(results),
            'failed_files': failed_files,
            'results': results,
            'analysis_config': {
                'fit_method': self.config.fit_method,
                'enable_gmm': self.config.enable_gmm,
                'gmm_log_transform': self.config.gmm_log_transform
            }
        }
        
        # 保存批量分析报告
        self.file_manager.save_comprehensive_report(batch_summary, self.config)
        
        print(f"\n批量分析完成!")
        print(f"成功: {len(results)}/{len(file_list)} 个文件")
        if failed_files:
            print(f"失败的文件: {failed_files}")
        
        return batch_summary


def create_argument_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(description='重构后的时钟同步分析工具')
    
    # 基本参数
    parser.add_argument('input', nargs='*', help='输入日志文件或目录')
    parser.add_argument('--output_dir', default='results/refactored_analysis', 
                       help='输出目录 (默认: results/refactored_analysis)')
    
    # 数据过滤参数
    parser.add_argument('--lower_percent', type=float, default=0.01, 
                       help='下分位过滤百分比 (默认: 0.01)')
    parser.add_argument('--upper_percent', type=float, default=0.99, 
                       help='上分位过滤百分比 (默认: 0.99)')
    
    # 拟合参数
    parser.add_argument('--fit_method', choices=['svr', 'lsq'], default='svr',
                       help='拟合方法 (默认: svr)')
    parser.add_argument('--piecewise_fit', action='store_true',
                       help='使用分段拟合')
    parser.add_argument('--piece_num', type=int, default=3,
                       help='分段数 (默认: 3)')
    parser.add_argument('--fit_lower_percent', type=float, default=0.01,
                       help='拟合数据下分位 (默认: 0.01)')
    parser.add_argument('--fit_upper_percent', type=float, default=0.3,
                       help='拟合数据上分位 (默认: 0.3)')
    
    # GMM参数
    parser.add_argument('--disable_gmm', action='store_true',
                       help='禁用GMM拟合')
    parser.add_argument('--gmm_min_components', type=int, default=1,
                       help='GMM最小组件数 (默认: 1)')
    parser.add_argument('--gmm_max_components', type=int, default=6,
                       help='GMM最大组件数 (默认: 6)')
    parser.add_argument('--no_gmm_log_transform', action='store_true',
                       help='GMM拟合时不使用对数变换')
    
    # 绘图参数
    parser.add_argument('--plot_real_time', action='store_true',
                       help='绘制实时时间序列图')
    parser.add_argument('--plot_raw_delay', action='store_true',
                       help='绘制原始时延图')
    parser.add_argument('--show_plots', action='store_true',
                       help='显示图表')
    
    # 其他参数
    parser.add_argument('--batch_mode', action='store_true',
                       help='批量处理模式')
    parser.add_argument('--no_save_details', action='store_true',
                       help='不保存详细结果数据')
    
    return parser


def main():
    """主函数"""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # 处理输入文件
    input_files = []
    if args.input:
        for path in args.input:
            if os.path.isfile(path):
                input_files.append(path)
            elif os.path.isdir(path):
                # 搜索目录中的.log文件
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith('.log'):
                            input_files.append(os.path.join(root, file))
    
    if not input_files:
        print("错误: 请提供有效的输入文件或目录")
        parser.print_help()
        return
    
    print(f"找到 {len(input_files)} 个日志文件")
    
    # 创建配置
    config = AnalysisConfig(
        lower_percent=args.lower_percent,
        upper_percent=args.upper_percent,
        fit_method=args.fit_method,
        fit_lower_percent=args.fit_lower_percent,
        fit_upper_percent=args.fit_upper_percent,
        piecewise_fit=args.piecewise_fit,
        piece_num=args.piece_num,
        enable_gmm=not args.disable_gmm,
        gmm_n_components_range=(args.gmm_min_components, args.gmm_max_components),
        gmm_log_transform=not args.no_gmm_log_transform,
        plot_real_time=args.plot_real_time,
        plot_raw_delay=args.plot_raw_delay,
        show_plots=args.show_plots,
        output_dir=args.output_dir,
        save_detailed_results=not args.no_save_details,
        batch_mode=args.batch_mode or len(input_files) > 1
    )
    
    # 创建分析器
    analyzer = ClockSyncAnalyzer(config)
    
    # 执行分析
    if config.batch_mode or len(input_files) > 1:
        analyzer.analyze_batch(input_files)
    else:
        analyzer.analyze_single_file(input_files[0])
    
    print(f"\n分析完成！结果保存在: {config.output_dir}")


if __name__ == '__main__':
    main()
