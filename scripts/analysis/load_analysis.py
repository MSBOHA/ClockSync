"""
负载分析脚本 - 使用重构后的工具分析CPU负载和网络负载
专门针对不同负载条件进行SVR拟合offset和GMM拟合时延分布
"""
import os
import sys
import pandas as pd
import json
from datetime import datetime

# 添加delayAnalyzer目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
delay_analyzer_dir = os.path.join(current_dir, 'delayAnalyzer')
sys.path.insert(0, delay_analyzer_dir)

from main import ClockSyncAnalyzer
from config import AnalysisConfig


def analyze_load_conditions():
    """分析不同负载条件下的时钟同步性能"""
    print("🔬 负载条件分析 - LSQ拟合 + GMM时延分布")
    print("=" * 80)    # 配置专门针对负载分析的参数
    config = AnalysisConfig(
        output_dir='results/load_analysis_lsq',  # 相对于项目根目录
        # LSQ拟合配置
        fit_method='lsq',
        piecewise_fit=True,
        piece_num=10,  # 10段拟合
        fit_lower_percent=0.01,
        fit_upper_percent=0.1,
        # GMM配置 - 固定8个组件
        enable_gmm=True,
        gmm_n_components_range=(8, 8),  # 固定使用8个组件
        gmm_log_transform=True,
        # 数据过滤
        lower_percent=0.005,
        upper_percent=0.995,
        # 其他设置
        batch_mode=True,
        save_detailed_results=True,
        plot_real_time=False
    )
    
    # 获取所有负载测试文件
    load_files = get_load_test_files()
    
    if not load_files:
        print("❌ 未找到负载测试文件")
        return    print(f"找到 {len(load_files)} 类负载测试文件")
    for category, files in load_files.items():
        print(f"  {category}: {len(files)} 个文件")
      # 执行批量分析
    analyzer = ClockSyncAnalyzer(config)
    
    all_results = {}
    
    for category, files in load_files.items():
        print(f"\n分析 {category} 负载...")
        
        try:
            category_results = analyzer.analyze_batch(files)
            all_results[category] = category_results
            
            if category_results and category_results.get('successful_files', 0) > 0:
                print(f"✅ {category} 完成: {category_results['successful_files']}/{category_results['total_files']} 成功")
            else:
                print(f"❌ {category} 分析失败")
        except Exception as e:
            print(f"❌ {category} 分析出错: {str(e)}")
            all_results[category] = None
    
    # 生成负载对比报告
    try:
        generate_load_comparison_report(all_results, config.output_dir)
        print(f"\n负载分析完成！结果保存在: {config.output_dir}")
    except Exception as e:
        print(f"❌ 报告生成失败: {str(e)}")
    
    return all_results


def get_load_test_files():
    """获取所有负载测试文件"""
    base_dir = r"c:\Git_Code\Clock_Sync\data\load_tests_0612"
    
    load_files = {
        'CPU负载': [],
        '网络流量负载': []
    }
    
    # CPU负载文件
    cpu_dir = os.path.join(base_dir, 'cpu_load', 'CPU负载')
    if os.path.exists(cpu_dir):
        for file in os.listdir(cpu_dir):
            if file.startswith('log') and file.endswith('.log') and 'original' in file:
                full_path = os.path.join(cpu_dir, file)
                load_files['CPU负载'].append(full_path)
                print(f"  找到CPU负载文件: {file}")
    else:
        print(f"❌ CPU负载目录不存在: {cpu_dir}")
    
    # 网络流量负载文件
    traffic_dir = os.path.join(base_dir, 'traffic_load', '流量负载')
    if os.path.exists(traffic_dir):
        for file in os.listdir(traffic_dir):
            if file.startswith('log') and file.endswith('.log') and 'original' in file:
                full_path = os.path.join(traffic_dir, file)
                load_files['网络流量负载'].append(full_path)
                print(f"  找到网络负载文件: {file}")
    else:
        print(f"❌ 网络负载目录不存在: {traffic_dir}")
    
    # 过滤掉空的类别
    load_files = {k: v for k, v in load_files.items() if v}
    
    return load_files


def generate_load_comparison_report(all_results, output_dir):
    """生成负载对比报告"""
    print("\n生成负载对比报告...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    reports_dir = os.path.join(output_dir, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    report_data = {
        'analysis_timestamp': datetime.now().isoformat(),
        'summary': {},
        'detailed_results': {}
    }
    
    comparison_stats = []
    gmm_comparison = []
    
    for category, results in all_results.items():
        if not results or not results.get('results'):
            continue
        
        print(f"  处理 {category} 结果...")
        
        category_stats = {
            'category': category,
            'total_files': results['total_files'],
            'successful_files': results['successful_files'],
            'success_rate': results['successful_files'] / results['total_files'] if results['total_files'] > 0 else 0
        }
        
        # 收集拟合质量统计
        fit_rmse_list = []
        fit_r2_list = []
        data_points_list = []
        
        # 收集GMM统计
        gmm_stats = {
            '上行时延': {'bic_list': [], 'aic_list': [], 'means_list': []},
            '下行时延': {'bic_list': [], 'aic_list': [], 'means_list': []}
        }
        
        for result in results['results']:
            load_level = result.get('load_level', 'unknown')
            data_points = result.get('data_points', 0)
            fit_quality = result.get('fit_quality', {})
            gmm_results = result.get('gmm_results', {})
            data_points_list.append(data_points)
            
            if fit_quality.get('rmse') is not None:
                fit_rmse_list.append(fit_quality['rmse'])
            if fit_quality.get('r2') is not None:
                fit_r2_list.append(fit_quality['r2'])
              # 收集GMM结果
            for delay_type in ['上行时延', '下行时延']:
                if delay_type in gmm_results:
                    gmm_info = gmm_results[delay_type]
                    # 检查GMM参数是否存在
                    if 'bic' in gmm_info:
                        gmm_stats[delay_type]['bic_list'].append(gmm_info['bic'])
                    if 'aic' in gmm_info:
                        gmm_stats[delay_type]['aic_list'].append(gmm_info['aic'])
                    if 'means' in gmm_info:
                        gmm_stats[delay_type]['means_list'].extend(gmm_info['means'])
                    
                    # 添加到GMM对比表
                    weights = gmm_info.get('weights', [])
                    means = gmm_info.get('means', [])
                    stds = gmm_info.get('stds', [])
                    
                    # 计算权重统计
                    weights_mean = sum(weights) / len(weights) if weights else float('nan')
                    weights_std = (sum((w - weights_mean)**2 for w in weights) / len(weights))**0.5 if len(weights) > 1 else 0.0
                    
                    # 为负载等级添加类型标注
                    if category == 'CPU负载':
                        load_label = f"CPU{load_level}"
                    elif category == '网络流量负载':
                        load_label = f"网络{load_level}M"
                    else:
                        load_label = str(load_level)
                    
                    gmm_comparison.append({
                        'category': category,
                        'load_level': load_level,
                        'load_label': load_label,
                        'delay_type': delay_type,
                        'n_components': gmm_info.get('n_components', 0),
                        'bic': gmm_info.get('bic', float('nan')),
                        'aic': gmm_info.get('aic', float('nan')),
                        'weights_mean': weights_mean,
                        'weights_std': weights_std,
                        'weights': weights,
                        'means': means,
                        'stds': stds
                    })
        
        # 计算类别统计量
        if fit_rmse_list:
            category_stats.update({
                'avg_rmse': sum(fit_rmse_list) / len(fit_rmse_list),
                'min_rmse': min(fit_rmse_list),
                'max_rmse': max(fit_rmse_list)
            })
        
        if fit_r2_list:
            category_stats.update({
                'avg_r2': sum(fit_r2_list) / len(fit_r2_list),
                'min_r2': min(fit_r2_list),
                'max_r2': max(fit_r2_list)
            })
        
        if data_points_list:
            category_stats.update({
                'avg_data_points': sum(data_points_list) / len(data_points_list),
                'total_data_points': sum(data_points_list)
            })
        
        # GMM统计
        for delay_type, stats in gmm_stats.items():
            if stats['bic_list']:
                category_stats[f'{delay_type}_avg_bic'] = sum(stats['bic_list']) / len(stats['bic_list'])
                category_stats[f'{delay_type}_avg_aic'] = sum(stats['aic_list']) / len(stats['aic_list'])
        
        comparison_stats.append(category_stats)
        report_data['detailed_results'][category] = results
      # 保存对比统计表
    if comparison_stats:
        comparison_df = pd.DataFrame(comparison_stats)
        comparison_path = os.path.join(output_dir, 'reports', 'load_comparison_summary.csv')
        os.makedirs(os.path.dirname(comparison_path), exist_ok=True)
        comparison_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        print(f"负载对比统计已保存: {comparison_path}")
    
    # 保存GMM对比表
    if gmm_comparison:
        gmm_df = pd.DataFrame(gmm_comparison)
        gmm_path = os.path.join(output_dir, 'reports', 'gmm_comparison_detailed.csv')
        gmm_df.to_csv(gmm_path, index=False, encoding='utf-8-sig')
        print(f"GMM对比详情已保存: {gmm_path}")
    
    # 保存完整报告
    report_data['summary'] = comparison_stats
    report_path = os.path.join(output_dir, 'reports', 'load_analysis_comprehensive_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"完整报告已保存: {report_path}")
    
    # 打印摘要
    print_load_analysis_summary(comparison_stats, gmm_comparison)


def print_load_analysis_summary(comparison_stats, gmm_comparison):
    """打印负载分析摘要"""
    print("\n负载分析摘要")
    print("=" * 60)
    
    if comparison_stats:
        print("拟合质量对比:")
        print(f"{'负载类型':<15} {'成功率':<8} {'平均RMSE':<12} {'平均R²':<10}")
        print("-" * 50)
        
        for stats in comparison_stats:
            success_rate = f"{stats.get('success_rate', 0)*100:.1f}%"
            avg_rmse = f"{stats.get('avg_rmse', 0):.2f}" if stats.get('avg_rmse') else "N/A"
            avg_r2 = f"{stats.get('avg_r2', 0):.3f}" if stats.get('avg_r2') else "N/A"
            
            print(f"{stats['category']:<15} {success_rate:<8} {avg_rmse:<12} {avg_r2:<10}")
    
    if gmm_comparison:
        print(f"\nGMM拟合结果 (8组件):")
        
        # 按负载类型和时延类型分组
        gmm_by_category = {}
        for gmm in gmm_comparison:
            category = gmm['category']
            delay_type = gmm['delay_type']
            key = f"{category}_{delay_type}"
            
            if key not in gmm_by_category:
                gmm_by_category[key] = []
            gmm_by_category[key].append(gmm)
        
        print(f"{'负载类型':<15} {'时延类型':<10} {'平均BIC':<12} {'平均AIC':<12}")
        print("-" * 50)
        
        for key, gmm_list in gmm_by_category.items():
            if gmm_list:
                category = gmm_list[0]['category']
                delay_type = gmm_list[0]['delay_type']
                avg_bic = sum(g['bic'] for g in gmm_list) / len(gmm_list)
                avg_aic = sum(g['aic'] for g in gmm_list) / len(gmm_list)
                
                print(f"{category:<15} {delay_type:<10} {avg_bic:<12.2f} {avg_aic:<12.2f}")
    
    print("\n分析建议:")
    if comparison_stats:
        best_rmse = min(comparison_stats, key=lambda x: x.get('avg_rmse', float('inf')))
        best_r2 = max(comparison_stats, key=lambda x: x.get('avg_r2', 0))
        
        print(f"  最佳拟合质量(RMSE): {best_rmse['category']}")
        print(f"  最佳拟合质量(R²): {best_r2['category']}")
    print(f"  使用SVR分段拟合(10段)和GMM拟合(8组件)")
    print(f"  详细结果已保存到输出目录")


def analyze_specific_loads(load_levels=None):
    """分析特定负载等级"""
    if load_levels is None:
        load_levels = ['100M', '200M', '400M', 'cpu1', 'cpu2', 'cpu4']
    
    print(f"\n🎯 特定负载等级分析: {load_levels}")
    print("=" * 60)
    
    load_files = get_load_test_files()
    specific_files = []
    for category, files in load_files.items():
        for file in files:
            for level in load_levels:
                if level in os.path.basename(file):
                    specific_files.append(file)
                    print(f"  📁 {os.path.basename(file)} -> {level}")
                    break
    
    if not specific_files:
        print("❌ 未找到指定负载等级的文件")
        return
    
    # 专门的配置
    config = AnalysisConfig(
        output_dir='../../results/load_analysis_lsq',  # 输出到根目录的results下
        fit_method='lsq',
        piecewise_fit=True,
        fit_lower_percent=0.01,
        fit_upper_percent=0.1,
        piece_num=10,
        enable_gmm=True,
        gmm_n_components_range=(8, 8),
        gmm_log_transform=True,
        save_detailed_results=True
    )
    
    analyzer = ClockSyncAnalyzer(config)
    results = analyzer.analyze_batch(specific_files)
    
    if results:
        print(f"\n✅ 特定负载分析完成: {results['successful_files']}/{results['total_files']}")
        return results
    else:
        print("❌ 特定负载分析失败")
        return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='负载条件分析工具')
    parser.add_argument('--specific_loads', nargs='*', 
                       help='指定要分析的负载等级 (如: 100M 200M cpu1 cpu2)')
    
    args = parser.parse_args()
    
    print("负载条件分析工具 - SVR分段拟合 + GMM时延分布")
    print("=" * 60)
    
    # 全面负载分析
    print("执行负载分析...")
    all_results = analyze_load_conditions()
    
    # 特定负载分析 (如果指定)
    if args.specific_loads:
        print("\n执行特定负载分析...")
        specific_results = analyze_specific_loads(args.specific_loads)
    
    print("\n负载分析完成！")
    print("查看 results/load_analysis_lsq/ 目录获取完整结果")


if __name__ == '__main__':
    main()
