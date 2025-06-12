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

    numeric_lines_data = [] 
    single_column_error_values = [] 
    
    lines_read_for_type_detection = 0
    potential_column_counts_numeric_lines = [] 
    header_lines_count = 0 

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()

        if not all_lines:
            return "empty", None

        first_numeric_line_index = -1

        for idx, line_content in enumerate(all_lines):
            stripped_line = line_content.strip()
            if not stripped_line or stripped_line.startswith("//") or stripped_line.startswith("#"):
                if first_numeric_line_index == -1: 
                    header_lines_count +=1
                continue

            parts = stripped_line.split()
            
            if len(parts) == 1:
                try:
                    val = float(parts[0])
                    if first_numeric_line_index == -1 and not potential_column_counts_numeric_lines:
                         first_numeric_line_index = idx 
                    single_column_error_values.append(val)
                    if lines_read_for_type_detection < sample_lines_for_type_detection:
                        potential_column_counts_numeric_lines.append(1) 
                except ValueError:
                    if first_numeric_line_index == -1: 
                        header_lines_count +=1
            elif len(parts) == 2: 
                try:
                    time_val = float(parts[0])
                    error_val = float(parts[1])
                    if first_numeric_line_index == -1:
                        first_numeric_line_index = idx 
                    if lines_read_for_type_detection < sample_lines_for_type_detection:
                        potential_column_counts_numeric_lines.append(2)
                    numeric_lines_data.append({'time': time_val, 'error': error_val})
                except ValueError:
                    if first_numeric_line_index == -1: 
                        header_lines_count +=1
            else: 
                if first_numeric_line_index == -1:
                    header_lines_count +=1
            
            if first_numeric_line_index != -1: 
                 lines_read_for_type_detection += 1
                 if lines_read_for_type_detection >= sample_lines_for_type_detection:
                     break
        
        log_type = "unknown"
        avg_cols_numeric = 0
        if potential_column_counts_numeric_lines:
            avg_cols_numeric = sum(potential_column_counts_numeric_lines) / len(potential_column_counts_numeric_lines)

        if avg_cols_numeric > 1.5 and avg_cols_numeric < 2.5:
            log_type = "normal"
            # 对于 normal 类型，重新完整提取数据，而不是只用类型检测时收集的
            full_normal_data = []
            if first_numeric_line_index != -1: # 确保知道数据从哪里开始
                for line_content in all_lines[first_numeric_line_index:]:
                    stripped_line = line_content.strip()
                    if not stripped_line or stripped_line.startswith("//") or stripped_line.startswith("#"):
                        continue
                    parts = stripped_line.split()
                    if len(parts) == 2:
                        try:
                            time_val = float(parts[0])
                            error_val = float(parts[1])
                            full_normal_data.append({'time': time_val, 'error': error_val})
                        except ValueError:
                            # 在数据区如果遇到无法转换的双列，可以选择停止或跳过
                            pass 
            
            if not full_normal_data: 
                # 如果完整提取后还是没有数据，但类型检测时有（不太可能发生在此逻辑下，但作为保险）
                if numeric_lines_data: # numeric_lines_data 是类型检测时收集的
                     return "normal", pd.DataFrame(numeric_lines_data) 
                return "unknown", None
            return "normal", pd.DataFrame(full_normal_data)

        elif single_column_error_values and (avg_cols_numeric > 0.5 and avg_cols_numeric < 1.5):
            log_type = "error_single_column"
        elif not potential_column_counts_numeric_lines and single_column_error_values: 
            log_type = "error_single_column"

        if log_type == "error_single_column":
            final_error_values = []
            if first_numeric_line_index != -1: 
                for line_content in all_lines[first_numeric_line_index:]:
                    stripped_line = line_content.strip()
                    if not stripped_line or stripped_line.startswith("//") or stripped_line.startswith("#"):
                        continue
                    parts = stripped_line.split()
                    if len(parts) == 1: 
                        try:
                            error_val = float(parts[0])
                            final_error_values.append(error_val)
                        except ValueError:
                            pass 
            
            if not final_error_values and single_column_error_values: 
                final_error_values = single_column_error_values

            if not final_error_values:
                return "unknown", None 
            
            df = pd.DataFrame({'error': final_error_values})
            df['time'] = range(len(df)) 
            return "error_single_column", df
        
        if first_numeric_line_index == -1 and header_lines_count > 0 and not single_column_error_values and not numeric_lines_data:
            return "empty", None 

        return "unknown", None

    except Exception as e:
        print(f"  Error during type identification or data extraction for {filepath}: {e}")
        return "unknown", None


def calculate_stats_from_log(filepath, label, time_range_filter=None, skip_rows_for_single_col=10): # 新增参数
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

    # --- 修改：通用化跳过行数的处理 ---
    if log_type == "error_single_column" and skip_rows_for_single_col > 0:
        if len(df) > skip_rows_for_single_col:
            print(f"  Info: For {label} (single_column type), skipping first {skip_rows_for_single_col} rows.")
            df = df.iloc[skip_rows_for_single_col:].copy()
            df.reset_index(drop=True, inplace=True)
            df['time'] = range(len(df)) 
        else:
            print(f"  Warning: For {label} (single_column type), data points ({len(df)}) are less than or equal to rows to skip ({skip_rows_for_single_col}). No rows skipped or data might be empty after skip.")
    # --- 结束跳过行数处理 ---

    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['error'] = pd.to_numeric(df['error'], errors='coerce')    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['time', 'error'], inplace=True)

    if df.empty:
        print(f"  No valid numeric data after conversion or special processing for {label}.")
        return None
    
    if log_type == "error_single_column" and len(df) > MAX_POINTS_FOR_ERROR_LOG:
        print(f"  Info: {label} (error_single_column type) has {len(df)} points. Truncating to first {MAX_POINTS_FOR_ERROR_LOG} points.")
        df = df.head(MAX_POINTS_FOR_ERROR_LOG).copy() 
    
    if time_range_filter:
        start_t, end_t = time_range_filter
        condition = True 
        time_series = df['time'] if isinstance(df['time'], pd.Series) else pd.Series([df['time']])
        if start_t is not None:
            condition &= (time_series >= start_t) 
        if end_t is not None:
            condition &= (time_series <= end_t)
        if not (isinstance(condition, bool) and condition is True):
            df = df[condition].copy() 
        if df.empty:
            print(f"  No data after time range filtering for {label}.")
            return None

    if df.empty: 
        print(f"  No data to process for {label} after all steps.")
        return None

    if df.empty or len(df['time']) == 0:
        print(f"  No time data to adjust for {label}.")
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


def main(top_n_to_plot=5, rows_to_skip_default=0): # 新增参数 rows_to_skip_default
    """
    主函数：查找日志文件，处理它们，排序并生成图表和统计表。
    :param top_n_to_plot: 要绘制误差变化图的最佳参数组合数量。
    :param rows_to_skip_default: 对于 error_single_column 类型日志，默认跳过的起始行数。
    """
    log_files = glob.glob("../../data/phase_deviation_0524/timeError_*.log")

    if not log_files:
        print("No 'timeError_*.log' files found in the target data directory.") # 更正了提示信息
        return

    all_results = []
    print("Starting processing of timeError log files...")
    for file_path in log_files:
        filename = os.path.basename(file_path)
        
        match = re.search(r"timeError_((?:a\d_b\d)(?:_[^_20\.]+)?(?:_sync\d+)?)_", filename, re.IGNORECASE)
        if match:
            label = match.group(1)
        else: 
            label = filename.replace("timeError_", "").split('_20')[0] 
            if not label: label = filename 

        current_time_range = None 
        
        # 根据需要决定为特定标签或所有 single_column 日志跳过多少行
        # 这里我们使用 main 函数传入的默认值
        # 如果需要更细致的控制，可以在这里加入基于 label 的判断
        rows_to_skip_for_current_file = rows_to_skip_default
        # 示例：如果想只为 a0_b2 跳过，可以这样做 (但您要求通用)
        # if "a0_b2" in label.lower():
        #     rows_to_skip_for_current_file = 10 # 或者其他特定值
        # else:
        #     rows_to_skip_for_current_file = 0

        stats_data = calculate_stats_from_log(file_path, label,
                                              time_range_filter=current_time_range,
                                              skip_rows_for_single_col=rows_to_skip_for_current_file) # 传递参数

        # 使用新的指标 (例如 rms_offset) 进行有效性检查
        if stats_data and stats_data['df'] is not None and not stats_data['df'].empty and pd.notna(stats_data['rms_offset']):
            all_results.append(stats_data)
        else:
            # 更新跳过信息的措辞，如果不再基于 MAO
            print(f"  Skipping {label} (File: {filename}) due to processing issues or no valid RMS Offset.")

    if not all_results:
        print("No data could be processed successfully from the log files.")
        return

    # 按新的指标 (例如 rms_offset) 排序
    all_results.sort(key=lambda x: x['rms_offset'])

    # 更新排序结果的打印标题和内容
    print("\n--- Sorted Results by RMS Offset (Lower is Better) ---")
    for res in all_results:
        print(f"  Label: {res['label']}, RMS: {res['rms_offset']:.3f} µs, MaxAbs: {res['max_abs_offset']:.3f} µs, File: {os.path.basename(res['filepath'])}")

    if all_results:
        stats_summary_data = []
        for res in all_results:
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
        print("\\n\\n--- Overall Statistics Summary ---")
        print(summary_df.to_string(index=False)) 
        
        summary_csv_path = "../../results/phase_deviation_0524/statistics_summary.csv"
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"\\nStatistics summary saved to {summary_csv_path}")
        except Exception as e:
            print(f"\\nError saving statistics summary to CSV: {e}")

    results_to_plot = all_results[:top_n_to_plot]
    
    if not results_to_plot:
        print(f"\nNo results to plot (top_n_to_plot might be 0 or no valid results after sorting).")
        return

    plt.figure(figsize=(18, 10))
    num_to_plot_actually = len(results_to_plot)

    if num_to_plot_actually <= 8:
        colors = plt.cm.get_cmap('Dark2', num_to_plot_actually if num_to_plot_actually > 0 else 1)
    elif num_to_plot_actually <= 9:
        colors = plt.cm.get_cmap('Set1', num_to_plot_actually if num_to_plot_actually > 0 else 1)
    else: 
        colors = plt.cm.get_cmap('tab10', num_to_plot_actually if num_to_plot_actually > 0 else 1)

    print(f"\nPlotting top {num_to_plot_actually} results based on RMS Offset...") # 更新绘图说明
    
    # Y轴调整代码已被移除

    for i, result in enumerate(results_to_plot):
        df_to_plot = result['df']
        # 更新绘图标签以反映排序依据
        plot_label = f"{result['label']} (RMS: {result['rms_offset']:.3f} µs)"
        
        color_val = colors(i % colors.N) if num_to_plot_actually > 0 else colors(0.5)

        plt.plot(df_to_plot['time_adjusted'], 
                 df_to_plot['error'], 
                 label=plot_label, 
                 color=color_val, 
                 marker='.', 
                 linestyle='-', 
                 markersize=2,   
                 linewidth=0.7,  
                 alpha=0.8)      

    plt.xlabel('Sample Index') 
    plt.ylabel('Time Offset (µs)') 
    title_str = f'Top {num_to_plot_actually} Time Offset Comparison (Sorted by RMS Offset)' # 更新图表标题
    plt.title(title_str) 
    
    if num_to_plot_actually > 0:
        if num_to_plot_actually > 10 : 
             plt.legend(loc='best', ncol=2, fontsize='small')
        else:
             plt.legend(loc='best')
        
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    output_svg_path = "../../results/phase_deviation_0524/time_offset_comparison_top_n_rms.svg" # 建议修改输出文件名以反映排序标准
    output_png_path = "../../results/phase_deviation_0524/time_offset_comparison_top_n_rms.png" # 建议修改输出文件名以反映排序标准
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_svg_path), exist_ok=True)
    
    plt.savefig(output_svg_path)
    plt.savefig(output_png_path)
    print(f"\\nPlots for top {num_to_plot_actually} results saved as {output_svg_path} and {output_png_path}")
    
    plt.show()

if __name__ == "__main__":
    # 现在可以传递一个默认的跳过行数给 main 函数
    # 如果大多数 single_column 文件需要跳过10行，可以这样设置：
    main(top_n_to_plot=10, rows_to_skip_default=10)
    # 如果不需要跳过任何行，设置为 0：
    # main(top_n_to_plot=10, rows_to_skip_default=0)