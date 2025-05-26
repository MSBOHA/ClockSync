import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re
import numpy as np # 确保导入 numpy

# 设置当前工作目录为脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 为错误类型日志（单列数据）设置最大数据点数量，以防数据过多影响绘图
MAX_POINTS_FOR_ERROR_LOG = 7000 # 如果单列错误日志点数超过此值，则裁剪

def identify_log_type_and_extract_data(filepath, sample_lines_for_type_detection=30):
    """
    通过分析内容来识别日志文件类型，并提取数据。
    返回: (log_type, data_frame)
          log_type可以是 "normal", "error_single_column", "unknown", "empty", "not_found"
          data_frame是提取的数据，对于error_single_column，time列是索引
    """
    if not os.path.exists(filepath):
        return "not_found", None

    numeric_lines_data_for_type_detection = [] # 用于类型检测时收集 normal 类型数据
    single_column_error_values_for_type_detection = [] # 用于类型检测时收集 single_column 数据
    
    lines_read_for_type_detection = 0
    potential_column_counts_numeric_lines = [] 
    header_lines_count = 0 
    first_numeric_line_index = -1

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()

        if not all_lines:
            return "empty", None

        # 阶段1: 类型检测扫描 + 初步数据收集
        for idx, line_content in enumerate(all_lines):
            stripped_line = line_content.strip()
            if not stripped_line or stripped_line.startswith("//") or stripped_line.startswith("#"):
                if first_numeric_line_index == -1: 
                    header_lines_count +=1
                continue

            parts = stripped_line.split()
            
            # 仅在类型检测阶段收集数据
            if lines_read_for_type_detection < sample_lines_for_type_detection:
                if len(parts) == 1:
                    try:
                        val = float(parts[0])
                        if first_numeric_line_index == -1: first_numeric_line_index = idx
                        single_column_error_values_for_type_detection.append(val)
                        potential_column_counts_numeric_lines.append(1)
                        lines_read_for_type_detection +=1
                    except ValueError:
                        if first_numeric_line_index == -1: header_lines_count +=1
                elif len(parts) == 2: 
                    try:
                        time_val = float(parts[0])
                        error_val = float(parts[1])
                        if first_numeric_line_index == -1: first_numeric_line_index = idx
                        numeric_lines_data_for_type_detection.append({'time': time_val, 'error': error_val})
                        potential_column_counts_numeric_lines.append(2)
                        lines_read_for_type_detection +=1
                    except ValueError:
                        if first_numeric_line_index == -1: header_lines_count +=1
                else:
                    if first_numeric_line_index == -1: header_lines_count +=1
            elif first_numeric_line_index == -1 : # 如果超过了类型检测行数但还没找到数字行
                 if len(parts) == 1: # 尝试找到第一个数字行
                    try: float(parts[0]); first_numeric_line_index = idx; break
                    except ValueError: pass
                 elif len(parts) == 2:
                    try: float(parts[0]); float(parts[1]); first_numeric_line_index = idx; break
                    except ValueError: pass
        
        if first_numeric_line_index == -1 and header_lines_count > 0 and not single_column_error_values_for_type_detection and not numeric_lines_data_for_type_detection:
             return "empty", None # 文件只有头部或注释

        log_type = "unknown"
        avg_cols_numeric = 0
        if potential_column_counts_numeric_lines:
            avg_cols_numeric = sum(potential_column_counts_numeric_lines) / len(potential_column_counts_numeric_lines)

        # 根据类型检测阶段的平均列数判断类型
        if avg_cols_numeric > 1.5 and avg_cols_numeric < 2.5:
            log_type = "normal"
        elif potential_column_counts_numeric_lines and (avg_cols_numeric > 0.5 and avg_cols_numeric < 1.5): # 确保有数字行被统计
            log_type = "error_single_column"
        elif not potential_column_counts_numeric_lines and single_column_error_values_for_type_detection: # 只有单列数据被初步收集
             log_type = "error_single_column"


        # 阶段 2: 基于判定的类型完整提取数据
        if log_type == "normal":
            full_normal_data = []
            start_index_to_read = first_numeric_line_index if first_numeric_line_index !=-1 else 0
            for line_content in all_lines[start_index_to_read:]:
                stripped_line = line_content.strip()
                if not stripped_line or stripped_line.startswith("//") or stripped_line.startswith("#"):
                    continue
                parts = stripped_line.split()
                if len(parts) == 2:
                    try:
                        time_val = float(parts[0])
                        error_val = float(parts[1])
                        full_normal_data.append({'time': time_val, 'error': error_val})
                    except ValueError: pass 
            if not full_normal_data: return "unknown", None
            return "normal", pd.DataFrame(full_normal_data)

        elif log_type == "error_single_column":
            final_error_values = []
            start_index_to_read = first_numeric_line_index if first_numeric_line_index !=-1 else 0
            for line_content in all_lines[start_index_to_read:]:
                stripped_line = line_content.strip()
                if not stripped_line or stripped_line.startswith("//") or stripped_line.startswith("#"):
                    continue
                parts = stripped_line.split()
                if len(parts) == 1: 
                    try:
                        error_val = float(parts[0])
                        final_error_values.append(error_val)
                    except ValueError: pass 
            if not final_error_values: return "unknown", None
            df = pd.DataFrame({'error': final_error_values})
            df['time'] = range(len(df)) 
            return "error_single_column", df
        
        return "unknown", None

    except Exception as e:
        print(f"  Error during type identification or data extraction for {filepath}: {e}")
        return "unknown", None


def calculate_stats_from_log(filepath, label, time_range_filter=None, skip_rows_for_single_col=0):
    """
    处理单个日志文件：识别类型、提取数据、筛选、计算统计数据。
    :param skip_rows_for_single_col: 对于 error_single_column 类型，跳过开头的行数。
    返回包含统计数据和DataFrame的字典，或在失败时返回None。
    """
    log_type, df = identify_log_type_and_extract_data(filepath)

    print(f"\nProcessing: {label} (File: {os.path.basename(filepath)}) - Detected type: {log_type}")

    if df is None or df.empty:
        if log_type not in ["empty", "not_found"]:
             print(f"  No data extracted or file is empty/not found for {label}.")
        return None

    if log_type == "error_single_column" and skip_rows_for_single_col > 0:
        if len(df) > skip_rows_for_single_col:
            print(f"  Info: For {label} (single_column type), skipping first {skip_rows_for_single_col} rows.")
            df = df.iloc[skip_rows_for_single_col:].copy()
            df.reset_index(drop=True, inplace=True)
            df['time'] = range(len(df)) 
        else:
            print(f"  Warning: For {label} (single_column type), data points ({len(df)}) are less than or equal to rows to skip ({skip_rows_for_single_col}). No rows skipped or data might be empty after skip.")

    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['error'] = pd.to_numeric(df['error'], errors='coerce')    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['time', 'error'], inplace=True)

    if df.empty:
        print(f"  No valid numeric data after conversion or special processing for {label}.")
        return None
    
    # 对 error_single_column 类型的数据进行裁剪（如果点数过多）
    # 注意：这个裁剪应该在跳过行之后，在统计计算之前
    if log_type == "error_single_column" and len(df) > MAX_POINTS_FOR_ERROR_LOG:
        print(f"  Info: {label} (error_single_column type) has {len(df)} points. Truncating to first {MAX_POINTS_FOR_ERROR_LOG} points.")
        df = df.head(MAX_POINTS_FOR_ERROR_LOG).copy() 
    
    if time_range_filter:
        start_t, end_t = time_range_filter
        condition = True 
        time_series = df['time'] if isinstance(df['time'], pd.Series) else pd.Series([df['time']])
        if start_t is not None: condition &= (time_series >= start_t) 
        if end_t is not None: condition &= (time_series <= end_t)
        if not (isinstance(condition, bool) and condition is True):
            df = df[condition].copy() 
        if df.empty:
            print(f"  No data after time range filtering for {label}.")
            return None

    if df.empty: 
        print(f"  No data to process for {label} after all steps.")
        return None

    if df.empty or len(df['time']) == 0: # 应该不会执行到这里如果上面 df.empty 检查了
        print(f"  No time data to adjust for {label}.") # 防御性代码
        return None
        
    df.loc[:, 'time_adjusted'] = df['time'] - df['time'].iloc[0]

    mean_offset = df['error'].mean() 
    std_dev_offset = df['error'].std() 
    mean_abs_offset = df['error'].abs().mean() 
    
    print(f"  --- Statistics for {label} ---")
    print(f"    Mean Offset: {mean_offset:.3f} µs") 
    print(f"    Mean Absolute Offset: {mean_abs_offset:.3f} µs") 
    print(f"    Std Dev of Offset: {std_dev_offset:.3f} µs") 
    print(f"    Data points used for stats: {len(df)}")

    return {
        'label': label,
        'filepath': filepath,
        'log_type': log_type,
        'df': df,
        'mean_offset': mean_offset, 
        'mean_abs_offset': mean_abs_offset, 
        'std_dev_offset': std_dev_offset 
    }


def main(top_n_to_plot=5, rows_to_skip_default=0):
    """
    主函数：查找日志文件，处理它们，排序并生成图表和统计表。
    :param top_n_to_plot: 要绘制误差变化图的最佳参数组合数量。
    :param rows_to_skip_default: 对于 error_single_column 类型日志，默认跳过的起始行数。
    """
    own_method_log_files = glob.glob("timeError_*.log") # 自有方法的日志
    chrony_log_file = "timeError.log" # Chrony 的日志文件名

    all_own_method_results = []
    chrony_result_data = None

    print("Starting processing of 'Own Method' timeError log files...")
    for file_path in own_method_log_files:
        filename = os.path.basename(file_path)
        match = re.search(r"timeError_((?:a\d_b\d)(?:_[^_20\.]+)?(?:_sync\d+)?)_", filename, re.IGNORECASE)
        if match:
            label = match.group(1)
        else: 
            label = filename.replace("timeError_", "").split('_20')[0] 
            if not label: label = filename 

        current_time_range = None 
        rows_to_skip_for_current_file = rows_to_skip_default

        stats_data = calculate_stats_from_log(file_path, label, 
                                              time_range_filter=current_time_range,
                                              skip_rows_for_single_col=rows_to_skip_for_current_file)
        
        if stats_data and stats_data['df'] is not None and not stats_data['df'].empty and pd.notna(stats_data['mean_abs_offset']):
            all_own_method_results.append(stats_data)
        else:
            print(f"  Skipping {label} (File: {filename}) due to processing issues or no valid MAO.")

    # 处理 Chrony 日志
    if os.path.exists(chrony_log_file):
        print(f"\nProcessing Chrony log file: {chrony_log_file}...")
        # Chrony 日志通常不需要跳过行，除非其格式特殊
        chrony_stats = calculate_stats_from_log(chrony_log_file, "Chrony", skip_rows_for_single_col=0) 
        if chrony_stats and chrony_stats['df'] is not None and not chrony_stats['df'].empty and pd.notna(chrony_stats['mean_abs_offset']):
            chrony_result_data = chrony_stats
        else:
            print(f"  Skipping Chrony log (File: {chrony_log_file}) due to processing issues or no valid MAO.")
    else:
        print(f"\nChrony log file '{chrony_log_file}' not found.")

    # 准备用于统计摘要和排序的数据
    results_for_summary = list(all_own_method_results)
    if chrony_result_data:
        results_for_summary.append(chrony_result_data)

    if not results_for_summary:
        print("No data could be processed successfully from any log files.")
        return

    results_for_summary.sort(key=lambda x: x['mean_abs_offset']) 

    print("\n\n--- Overall Statistics Summary (Sorted by MAO) ---")
    summary_stats_list = []
    for res in results_for_summary:
        summary_stats_list.append({
            'Label': res['label'],
            'File': os.path.basename(res['filepath']),
            'Mean Offset (µs)': f"{res['mean_offset']:.3f}" if pd.notna(res['mean_offset']) else "N/A",
            'Mean Absolute Offset (µs)': f"{res['mean_abs_offset']:.3f}" if pd.notna(res['mean_abs_offset']) else "N/A",
            'Std Dev of Offset (µs)': f"{res['std_dev_offset']:.3f}" if pd.notna(res['std_dev_offset']) else "N/A",
            'Data Points': len(res['df']) if res['df'] is not None else 0
        })
    summary_df = pd.DataFrame(summary_stats_list)
    print(summary_df.to_string(index=False)) 
    
    summary_csv_path = "statistics_summary.csv"
    try:
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\nStatistics summary saved to {summary_csv_path}")
    except Exception as e:
        print(f"\nError saving statistics summary to CSV: {e}")

    # 绘制 Top N 自有方法的结果
    if all_own_method_results:
        all_own_method_results.sort(key=lambda x: x['mean_abs_offset']) 
        results_to_plot_own = all_own_method_results[:top_n_to_plot]
        
        if results_to_plot_own:
            plt.figure(figsize=(18, 10))
            num_to_plot_actually = len(results_to_plot_own)
            colors_cmap = plt.cm.get_cmap('Dark2' if num_to_plot_actually <= 8 else ('Set1' if num_to_plot_actually <=9 else 'tab10'), 
                                     num_to_plot_actually if num_to_plot_actually > 0 else 1)

            print(f"\nPlotting top {num_to_plot_actually} 'Own Method' results based on MAO...")
            for i, result in enumerate(results_to_plot_own):
                df_to_plot = result['df']
                plot_label = f"{result['label']} (MAO: {result['mean_abs_offset']:.3f} µs)"
                color_val = colors_cmap(i % colors_cmap.N if colors_cmap.N > 0 else 0.5)
                plt.plot(df_to_plot['time_adjusted'], df_to_plot['error'], label=plot_label, color=color_val, 
                         marker='.', linestyle='-', markersize=2, linewidth=0.7, alpha=0.8)      

            plt.xlabel('Sample Index') 
            plt.ylabel('Time Offset (µs)') 
            plt.title(f'Top {num_to_plot_actually} Own Method Time Offset Comparison (Sorted by MAO)')
            plt.legend(loc='best', ncol=2 if num_to_plot_actually > 10 else 1, fontsize='small' if num_to_plot_actually > 10 else 'medium')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            output_svg_path = "time_offset_comparison_top_n_own_methods.svg" 
            output_png_path = "time_offset_comparison_top_n_own_methods.png" 
            plt.savefig(output_svg_path)
            plt.savefig(output_png_path)
            print(f"\nPlots for top {num_to_plot_actually} 'Own Method' results saved as {output_svg_path} and {output_png_path}")
            plt.show()
        else:
            print("\nNo 'Own Method' results to plot for Top N.")
    else:
        print("\nNo 'Own Method' results found to plot.")

    # 单独绘制 Chrony 的结果图
    if chrony_result_data:
        plt.figure(figsize=(18, 10))
        df_chrony = chrony_result_data['df']
        label_chrony = f"Chrony (MAO: {chrony_result_data['mean_abs_offset']:.3f} µs, File: {os.path.basename(chrony_result_data['filepath'])})"
        
        plt.plot(df_chrony['time_adjusted'], df_chrony['error'], label=label_chrony, color='green',
                 marker='.', linestyle='-', markersize=2, linewidth=0.7, alpha=0.8)
        
        plt.xlabel('Sample Index')
        plt.ylabel('Time Offset (µs)')
        plt.title('Chrony Time Offset')
        plt.legend(loc='best')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        chrony_plot_svg_path = "chrony_time_offset.svg"
        chrony_plot_png_path = "chrony_time_offset.png"
        plt.savefig(chrony_plot_svg_path)
        plt.savefig(chrony_plot_png_path)
        print(f"\nChrony plot saved as {chrony_plot_svg_path} and {chrony_plot_png_path}")
        plt.show()
    else:
        print("\nChrony data not available or failed to process, skipping Chrony-only plot.")


    # 绘制最佳自有方法 vs Chrony 的对比图
    best_own_method_for_comparison = None
    if all_own_method_results: # all_own_method_results 已经按 MAO 排序
        best_own_method_for_comparison = all_own_method_results[0]

    if best_own_method_for_comparison and chrony_result_data:
        plt.figure(figsize=(18, 10))
        
        # 绘制最佳自有方法
        df_best_own = best_own_method_for_comparison['df']
        label_best_own = f"Best Own: {best_own_method_for_comparison['label']} (MAO: {best_own_method_for_comparison['mean_abs_offset']:.3f} µs)"
        plt.plot(df_best_own['time_adjusted'], df_best_own['error'], label=label_best_own, color='blue',
                 marker='.', linestyle='-', markersize=2, linewidth=0.7, alpha=0.8)
                 
        # 绘制 Chrony
        df_chrony = chrony_result_data['df']
        label_chrony = f"Chrony (MAO: {chrony_result_data['mean_abs_offset']:.3f} µs)"
        plt.plot(df_chrony['time_adjusted'], df_chrony['error'], label=label_chrony, color='red',
                 marker='.', linestyle='-', markersize=2, linewidth=0.7, alpha=0.8)

        plt.xlabel('Sample Index')
        plt.ylabel('Time Offset (µs)')
        plt.title('Comparison: Best Own Method vs. Chrony')
        plt.legend(loc='best')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        comp_svg_path = "comparison_best_own_vs_chrony.svg"
        comp_png_path = "comparison_best_own_vs_chrony.png"
        plt.savefig(comp_svg_path)
        plt.savefig(comp_png_path)
        print(f"\nComparison plot saved as {comp_svg_path} and {comp_png_path}")
        plt.show()
    elif not best_own_method_for_comparison:
        print("\nCannot create comparison plot: No best 'Own Method' result found.")
    elif not chrony_result_data:
        print("\nCannot create comparison plot: Chrony data not available or failed to process.")


if __name__ == "__main__":
    main(top_n_to_plot=5, rows_to_skip_default=0) # 默认不跳过行
    # main(top_n_to_plot=5, rows_to_skip_default=10) # 如果自有方法日志通常需要跳过10行