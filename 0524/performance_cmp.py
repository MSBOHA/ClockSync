import os
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re

# 设置当前工作目录为脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 为错误类型日志（单列数据）设置最大数据点数量，以防数据过多影响绘图
MAX_POINTS_FOR_ERROR_LOG = 600 # 如果单列错误日志点数超过此值，则裁剪

def identify_log_type_and_extract_data(filepath, sample_lines_for_type_detection=30):
    """
    通过分析内容来识别日志文件类型，并提取数据。
    返回: (log_type, data_frame)
          log_type可以是 "normal", "error_single_column", "unknown", "empty", "not_found"
          data_frame是提取的数据，对于error_single_column，time列是索引
    """
    if not os.path.exists(filepath):
        return "not_found", None

    numeric_lines_data = [] # 存储解析出的数字行
    text_lines_count = 0    # 存储非数字、非注释的文本行数量
    
    lines_read_for_type = 0
    potential_column_counts = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()

        # 阶段1: 类型检测 (基于文件前部内容)
        for line_content in all_lines:
            if lines_read_for_type >= sample_lines_for_type_detection:
                break
            
            stripped_line = line_content.strip()
            if not stripped_line or stripped_line.startswith("//") or stripped_line.startswith("#"):
                continue # 跳过空行和注释

            lines_read_for_type += 1
            parts = stripped_line.split()
            
            try:
                # 尝试将所有部分转换为数字
                [float(p) for p in parts]
                potential_column_counts.append(len(parts))
            except ValueError:
                # 如果转换失败，则认为是文本说明行
                text_lines_count +=1 

        log_type = "unknown"
        avg_cols = 0
        if potential_column_counts:
            avg_cols = sum(potential_column_counts) / len(potential_column_counts)

        if not lines_read_for_type and not text_lines_count : 
            is_truly_empty = True
            for line_content in all_lines:
                stripped_line = line_content.strip()
                if stripped_line and not stripped_line.startswith("//") and not stripped_line.startswith("#"):
                    is_truly_empty = False
                    break
            if is_truly_empty:
                return "empty", None

        if avg_cols > 1.5 and avg_cols < 2.5: 
            log_type = "normal"
        elif avg_cols > 0.5 and avg_cols < 1.5: 
            log_type = "error_single_column"
        elif text_lines_count > 0 and not potential_column_counts : 
            log_type = "error_single_column" 
        
        # 阶段2: 数据提取 (基于已判断的类型)
        if log_type == "normal":
            for line_content in all_lines:
                stripped_line = line_content.strip()
                if not stripped_line or stripped_line.startswith("//") or stripped_line.startswith("#"):
                    continue
                parts = stripped_line.split()
                if len(parts) == 2:
                    try:
                        time_val = float(parts[0])
                        error_val = float(parts[1])
                        numeric_lines_data.append({'time': time_val, 'error': error_val})
                    except ValueError:
                        pass 
            if not numeric_lines_data: return "unknown", None 
            return "normal", pd.DataFrame(numeric_lines_data)

        elif log_type == "error_single_column":
            error_values = []
            for line_content in all_lines:
                stripped_line = line_content.strip()
                if not stripped_line or stripped_line.startswith("//") or stripped_line.startswith("#"):
                    continue
                try:
                    error_val = float(stripped_line)
                    error_values.append(error_val)
                except ValueError:
                    pass 
            if not error_values: return "error_single_column", pd.DataFrame({'time': [], 'error': []}) 
            
            df = pd.DataFrame({'error': error_values})
            df['time'] = range(len(df)) 
            return "error_single_column", df
        
        else: 
            if log_type == "empty": return "empty", None
            return "unknown", None

    except Exception as e:
        print(f"  Error during type identification or data extraction for {filepath}: {e}")
        return "unknown", None


def process_and_plot_log_file(filepath, label, color, time_range_filter=None):
    """
    处理单个日志文件：识别类型、提取数据、筛选、绘图和统计。
    """
    log_type, df = identify_log_type_and_extract_data(filepath)

    print(f"\nProcessing: {label} (File: {os.path.basename(filepath)}) - Detected type: {log_type}")

    if df is None or df.empty:
        if log_type not in ["empty", "not_found"]:
             print(f"  No data extracted or file is empty/not found for {label}.")
        return

    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['error'] = pd.to_numeric(df['error'], errors='coerce')
    df.dropna(subset=['time', 'error'], inplace=True)

    if df.empty:
        print(f"  No valid numeric data after conversion for {label}.")
        return
    
    # 对 error_single_column 类型的数据进行裁剪（如果点数过多）
    if log_type == "error_single_column" and len(df) > MAX_POINTS_FOR_ERROR_LOG:
        print(f"  Info: {label} (error_single_column type) has {len(df)} points. Truncating to first {MAX_POINTS_FOR_ERROR_LOG} points.")
        df = df.head(MAX_POINTS_FOR_ERROR_LOG).copy()
    
    # 应用时间范围过滤器 (基于原始时间或索引)
    if time_range_filter:
        start_t, end_t = time_range_filter
        condition = True
        if start_t is not None:
            condition &= (df['time'] >= start_t) 
        if end_t is not None:
            condition &= (df['time'] <= end_t)
        df = df[condition].copy()
        if df.empty:
            print(f"  No data after time range filtering for {label}.")
            return

    if df.empty: 
        print(f"  No data to process for {label} after all steps.")
        return

    df.loc[:, 'time_adjusted'] = df['time'] - df['time'].iloc[0]

    plt.plot(df['time_adjusted'], df['error'], label=label, color=color, marker='.', linestyle='-', markersize=3)

    mean_error = df['error'].mean()
    std_error = df['error'].std()
    mean_abs_error = df['error'].abs().mean() # 计算平均绝对误差
    
    print(f"  --- Statistics for {label} ---")
    print(f"    Mean Error: {mean_error:.3f}")
    print(f"    Mean Absolute Error: {mean_abs_error:.3f}") # 输出平均绝对误差
    print(f"    Std Dev of Error: {std_error:.3f}")
    print(f"    Data points plotted: {len(df)}")


def main():
    """
    主函数：查找日志文件，处理它们，并生成图表。
    """
    plt.figure(figsize=(18, 10))

    log_files = glob.glob("timeError_*.log")
    log_files.sort()

    if not log_files:
        print("No 'timeError_*.log' files found in the current directory.")
        return

    num_files = len(log_files)
    colors = plt.cm.get_cmap('viridis', num_files if num_files > 0 else 1)

    print("Starting processing of timeError log files...")
    for i, file_path in enumerate(log_files):
        filename = os.path.basename(file_path)
        
        match = re.search(r"timeError_(a\d_b\d(?:_?[^_\d]*)?)_", filename, re.IGNORECASE)
        if match:
            label = match.group(1)
        else:
            label = filename.replace("timeError_", "").split('_20')[0] 
            if not label: label = filename 

        current_time_range = None 

        process_and_plot_log_file(file_path, label, colors(i / num_files if num_files > 1 else 0.5), 
                                  time_range_filter=current_time_range)

    plt.xlabel('Time (Adjusted, relative to start of each dataset)')
    plt.ylabel('Time Error')
    plt.title('Time Error Comparison for Different Parameters (Content-based Type Detection)')
    
    if num_files > 10:
        plt.legend(loc='best', ncol=2, fontsize='small')
    elif num_files > 0 : 
        plt.legend(loc='best')
        
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    output_svg_path = "time_error_comparison_content_detection_v2.svg" # 更新输出文件名
    output_png_path = "time_error_comparison_content_detection_v2.png" # 更新输出文件名
    plt.savefig(output_svg_path)
    plt.savefig(output_png_path)
    print(f"\nPlots saved as {output_svg_path} and {output_png_path}")
    
    plt.show()

if __name__ == "__main__":
    main()