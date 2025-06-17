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
    # mean_abs_offset = df['error'].abs().mean() # 原来的 Mean Absolute Offset 计算
    max_abs_offset = df['error'].abs().max()    # 新增：最大绝对值
    rms_offset = np.sqrt(np.mean(df['error']**2)) # 新增：RMS
    std_dev_offset = df['error'].std()          # 保留：标准差

    print(f"  --- Statistics for {label} ---")
    print(f"    Mean Offset: {mean_offset:.3f} µs")
    print(f"    Max Absolute Offset: {max_abs_offset:.3f} µs") # 更改：替换原来的 Mean Absolute Offset
    print(f"    RMS of Offset: {rms_offset:.3f} µs")         # 新增
    print(f"    Std Dev of Offset: {std_dev_offset:.3f} µs")
    print(f"    Data points used for stats: {len(df)}")

    return {
        'label': label,
        'filepath': filepath,
        'log_type': log_type,
        'df': df,
        'mean_offset': mean_offset,
        # 'mean_abs_offset': mean_abs_offset, # mean_abs_offset 不再返回
        'max_abs_offset': max_abs_offset,     # 新增返回
        'rms_offset': rms_offset,             # 新增返回
        'std_dev_offset': std_dev_offset
    }


def main(top_n_to_plot=9, rows_to_skip_default=0):
    """
    主函数：查找日志文件，处理它们，排序并生成图表和统计表。
    :param top_n_to_plot: 要绘制误差变化图的最佳参数组合数量。
    :param rows_to_skip_default: 对于 error_single_column 类型日志，默认跳过的起始行数。
    """
    all_own_method_results = []
    chrony_result_data = None

    log_dir = "../../data/phase_deviation_0524/"
    chrony_log_basename = "timeError.log" # Chrony 日志的特定文件名
    chrony_log_full_path = os.path.abspath(os.path.join(log_dir, chrony_log_basename))

    all_log_files_in_dir = glob.glob(os.path.join(log_dir, "*.log")) # Get all .log files

    own_method_log_files = []
    found_chrony_log_path = None

    for file_path_str in all_log_files_in_dir:
        current_file_abs_path = os.path.abspath(file_path_str)
        current_file_basename = os.path.basename(current_file_abs_path)

        if current_file_abs_path == chrony_log_full_path:
            found_chrony_log_path = current_file_abs_path
        elif current_file_basename.startswith("timeError_"): # Ensure it's an "own method" log
            own_method_log_files.append(current_file_abs_path)

    if not own_method_log_files and not found_chrony_log_path:
        print(f"No 'timeError_*.log' files or '{chrony_log_basename}' found in the target data directory: {log_dir}")
        return

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
        
        if stats_data and stats_data['df'] is not None and not stats_data['df'].empty and pd.notna(stats_data['rms_offset']):
            all_own_method_results.append(stats_data)
        else:
            print(f"  Skipping {label} (File: {filename}) due to processing issues or no valid RMS Offset.")

    # 处理 Chrony 日志
    if found_chrony_log_path:
        print(f"\nProcessing Chrony log file: {os.path.basename(found_chrony_log_path)}...")
        # Chrony 日志通常不需要跳过行，除非其格式特殊
        chrony_stats = calculate_stats_from_log(found_chrony_log_path, "Chrony", skip_rows_for_single_col=0)
        if chrony_stats and chrony_stats['df'] is not None and not chrony_stats['df'].empty and pd.notna(chrony_stats['rms_offset']):
            chrony_result_data = chrony_stats
        else:
            print(f"  Skipping Chrony log (File: {os.path.basename(found_chrony_log_path)}) due to processing issues or no valid RMS Offset.")
    else: # If found_chrony_log_path is None
        print(f"\\nChrony log file '{chrony_log_basename}' not found in {log_dir}.")

    # 准备用于统计摘要和排序的数据
    results_for_summary = list(all_own_method_results)
    if chrony_result_data:
        results_for_summary.append(chrony_result_data)

    if not all_own_method_results and not chrony_result_data:
        print("No data could be processed successfully from any log files.")
        return

    # 按 RMS Offset 排序 (results_for_summary 用于生成总表，all_own_method_results 用于绘图和选最佳)
    if all_own_method_results:
        all_own_method_results.sort(key=lambda x: x['rms_offset'])
    if results_for_summary:
        results_for_summary.sort(key=lambda x: x['rms_offset'])

    print("\n--- Sorted Combined Results by RMS Offset (Lower is Better) ---")
    for res in results_for_summary:
        print(f"  Label: {res['label']}, RMS: {res['rms_offset']:.3f} µs, MaxAbs: {res['max_abs_offset']:.3f} µs, MeanOffset: {res['mean_offset']:.3f} µs, StdDev: {res['std_dev_offset']:.3f} µs, File: {os.path.basename(res['filepath'])}")

    if results_for_summary:
        stats_summary_data = []
        for res in results_for_summary: # 使用合并后的列表生成总表
            stats_summary_data.append({
                'Label': res['label'],
                'File': os.path.basename(res['filepath']),
                'Mean Offset (µs)': f"{res['mean_offset']:.3f}" if pd.notna(res['mean_offset']) else "N/A",
                'Max Absolute Offset (µs)': f"{res['max_abs_offset']:.3f}" if pd.notna(res['max_abs_offset']) else "N/A", # 新增
                'RMS of Offset (µs)': f"{res['rms_offset']:.3f}" if pd.notna(res['rms_offset']) else "N/A",           # 新增
                'Std Dev of Offset (µs)': f"{res['std_dev_offset']:.3f}" if pd.notna(res['std_dev_offset']) else "N/A",
                'Data Points': len(res['df']) if res['df'] is not None else 0
            })
        
        summary_df = pd.DataFrame(stats_summary_data)
        print("\n\n--- Overall Statistics Summary (Sorted by RMS Offset) ---")
        print(summary_df.to_string(index=False)) 
        
        output_results_dir = "../../results/phase_deviation_0524/"
        os.makedirs(output_results_dir, exist_ok=True)
        summary_csv_path = os.path.join(output_results_dir, "statistics_summary_rms.csv")
        try:
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"\nStatistics summary saved to {summary_csv_path}")
        except Exception as e:
            print(f"\nError saving statistics summary to CSV: {e}")

    # 绘制 Top N 自有方法的结果
    if all_own_method_results:
        # all_own_method_results 已经按 RMS 排序
        results_to_plot_own = all_own_method_results[:top_n_to_plot]
        
        if results_to_plot_own:
            plt.figure(figsize=(18, 10))
            num_to_plot_actually = len(results_to_plot_own)
            
            if num_to_plot_actually <= 8:
                colors_cmap = plt.get_cmap('Dark2', num_to_plot_actually if num_to_plot_actually > 0 else 1)
            elif num_to_plot_actually <= 9:
                colors_cmap = plt.get_cmap('Set1', num_to_plot_actually if num_to_plot_actually > 0 else 1)
            else:
                colors_cmap = plt.get_cmap('tab10', num_to_plot_actually if num_to_plot_actually > 0 else 1)

            print(f"\nPlotting top {num_to_plot_actually} 'Own Method' results based on RMS Offset...")
            for i, result in enumerate(results_to_plot_own):
                df_to_plot = result['df']
                plot_label = f"{result['label']} (RMS: {result['rms_offset']:.3f} µs)"
                color_val = colors_cmap(i % colors_cmap.N if colors_cmap.N > 0 else 0.5)
                plt.plot(df_to_plot['time_adjusted'], df_to_plot['error'], label=plot_label, color=color_val, 
                         marker='.', linestyle='-', markersize=2, linewidth=0.7, alpha=0.8)      
            #设置x轴和y轴标签大小
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlabel('Sample Index') 
            plt.ylabel('Time Offset (µs)') 
            
            plt.title(f'Top {num_to_plot_actually} Own Method Time Offset Comparison (Sorted by RMS Offset)', fontsize=20)
            plt.legend(loc='best', ncol=2 if num_to_plot_actually > 10 else 1, fontsize=20)
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            output_svg_path = os.path.join(output_results_dir, "time_offset_comparison_top_n_own_methods_rms.svg")
            output_png_path = os.path.join(output_results_dir, "time_offset_comparison_top_n_own_methods_rms.png")
            os.makedirs(os.path.dirname(output_svg_path), exist_ok=True)
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
        label_chrony = f"Chrony (RMS: {chrony_result_data['rms_offset']:.3f} µs, File: {os.path.basename(chrony_result_data['filepath'])})"
        
        plt.plot(df_chrony['time_adjusted'], df_chrony['error'], label=label_chrony, color='green',
                 marker='.', linestyle='-', markersize=2, linewidth=0.7, alpha=0.8)
        
        plt.xlabel('Sample Index')
        plt.ylabel('Time Offset (µs)')
        plt.title('Chrony Time Offset (Sorted by RMS Offset)', fontsize=16)
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        chrony_plot_svg_path = os.path.join(output_results_dir, "chrony_time_offset_rms.svg")
        chrony_plot_png_path = os.path.join(output_results_dir, "chrony_time_offset_rms.png")
        os.makedirs(os.path.dirname(chrony_plot_svg_path), exist_ok=True)
        plt.savefig(chrony_plot_svg_path)
        plt.savefig(chrony_plot_png_path)
        print(f"\nChrony plot saved as {chrony_plot_svg_path} and {chrony_plot_png_path}")
        plt.show()
    else:
        print("\nChrony data not available or failed to process, skipping Chrony-only plot.")


    # 绘制最佳自有方法 vs Chrony 的对比图
    best_own_method_for_comparison = None
    if all_own_method_results: # all_own_method_results 已经按 RMS 排序
        best_own_method_for_comparison = all_own_method_results[0]

    if best_own_method_for_comparison and chrony_result_data:
        plt.figure(figsize=(18, 10))
        
        # 绘制最佳自有方法
        df_best_own = best_own_method_for_comparison['df']
        label_best_own = f"Best Own: {best_own_method_for_comparison['label']} (RMSE: {best_own_method_for_comparison['rms_offset']:.3f} µs)"
        plt.plot(df_best_own['time_adjusted'], df_best_own['error'], label=label_best_own, color='blue',
                 marker='.', linestyle='-', markersize=2, linewidth=0.7, alpha=0.8)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        # 绘制 Chrony
        df_chrony = chrony_result_data['df']
        label_chrony = f"Chrony (RMSE: {chrony_result_data['rms_offset']:.3f} µs)"
        plt.plot(df_chrony['time_adjusted'], df_chrony['error'], label=label_chrony, color='red',
                 marker='.', linestyle='-', markersize=2, linewidth=0.7, alpha=0.8)

        plt.xlabel('Sample Index', fontsize=20)
        plt.ylabel('Time Offset (µs)', fontsize=20)
        plt.title('Comparison: Best Own Method (by RMSE) vs. Chrony (by RMSE)', fontsize=20)
        plt.legend(loc='best', fontsize=20)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()

        comp_svg_path = os.path.join(output_results_dir, "comparison_best_own_vs_chrony_rms.svg")
        comp_png_path = os.path.join(output_results_dir, "comparison_best_own_vs_chrony_rms.png")
        os.makedirs(os.path.dirname(comp_svg_path), exist_ok=True)
        plt.savefig(comp_svg_path)
        plt.savefig(comp_png_path)
        print(f"\nComparison plot saved as {comp_svg_path} and {comp_png_path}")
        plt.show()
    elif not best_own_method_for_comparison:
        print("\nCannot create comparison plot: No best 'Own Method' result found.")
    elif not chrony_result_data:
        print("\nCannot create comparison plot: Chrony data not available or failed to process.")

if __name__ == "__main__":
    main(top_n_to_plot=9, rows_to_skip_default=0) # 默认不跳过行
    # main(top_n_to_plot=5, rows_to_skip_default=10) # 如果自有方法日志通常需要跳过10行