import os
import pandas as pd
import matplotlib.pyplot as plt

# 定义文件路径
time_error_log_file_path_own = r".\0507\自有方法\timeError.log"
time_error_log_file_path_chrony = r".\0507\chrony\timeError.log"
ptp4l_log_file_path = r".\0507\自有方法\ptp4l.log"

# 定义一个函数来读取和处理 timeError.log 数据
def process_time_error_log(file_path, label, color, time_range=None):
    if os.path.exists(file_path):
        # 读取数据
        data = pd.read_csv(file_path, sep=" ", header=None)
        data.columns = ['time', 'error']

        # 将 time 列转换为数值类型
        data['time'] = pd.to_numeric(data['time'], errors='coerce')

        # 如果提供了时间范围，则进行筛选
        if time_range:
            filtered_data = data[(data['time'] > time_range[0]) & (data['time'] < time_range[1])]
        else:
            filtered_data = data

        # 调整时间轴：减去第一个时间值
        filtered_data['time'] -= filtered_data['time'].iloc[0]

        # 绘制筛选后的 error 随时间变化的图
        plt.plot(filtered_data['time'], filtered_data['error'], label=label, color=color)

        # 计算筛选后 error 的统计特性
        mean_error = filtered_data['error'].mean()
        std_error = filtered_data['error'].std()
        print(f"{label} - Filtered Mean Error: {mean_error}")
        print(f"{label} - Filtered Standard Deviation of Error: {std_error}")
    else:
        print(f"文件不存在: {file_path}")

# 定义一个函数来读取和处理 ptp4l.log 数据
def process_ptp4l_log(file_path, time_range=None):
    if os.path.exists(file_path):
        # 打开文件并逐行读取
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # 过滤包含 "master offset" 的行
        filtered_lines = [line for line in lines if "master offset" in line]

        # 提取 master offset 的值
        data = []
        for line in filtered_lines:
            parts = line.split()
            # 提取 [] 中的时间
            time = parts[0].split('[')[1].split(']')[0]
            # 提取 master offset 的值
            master_offset = parts[parts.index("master") + 2]
            # 提取 frequency
            frequency = parts[parts.index("freq") + 1]
            # 提取 delay
            delay = parts[parts.index("delay") + 1]
            # 转化为浮点数
            time = float(time)
            master_offset = float(master_offset)
            frequency = float(frequency)
            delay = float(delay)
            data.append([time, master_offset, frequency, delay])

        # 创建 DataFrame
        df = pd.DataFrame(data, columns=["Time", "Master Offset", "Frequency", "Delay"])
        # 调整时间轴：减去第一个时间值

        df["Time"] -= df["Time"].iloc[0]
        # 如果提供了时间范围，则进行筛选

        if time_range:
            df = df[(df["Time"] > time_range[0]) & (df["Time"] < time_range[1])]

        # 绘制 master offset 的图
        plt.plot(df["Time"], df["Master Offset"], label="PTP Master Offset", color='red')

        # 输出统计特性
        mean_offset = df["Master Offset"].mean()
        std_offset = df["Master Offset"].std()
        print(f"PTP - Mean Master Offset: {mean_offset}")
        print(f"PTP - Standard Deviation of Master Offset: {std_offset}")
    else:
        print(f"文件不存在: {file_path}")

# 创建图表
plt.figure(figsize=(12, 8))

# 处理自有方法的 timeError.log
process_time_error_log(
    time_error_log_file_path_own,
    label="Own Method",
    color="green",
    time_range=(5, 390)  # 可选的时间范围
)

# 处理 chrony 的 timeError.log
process_time_error_log(
    time_error_log_file_path_chrony,
    label="Chrony Method",
    color="blue",
    time_range=None  # 可选的时间范围
)

# 处理 ptp4l.log
process_ptp4l_log(ptp4l_log_file_path, time_range=(5, 390))  # 可选的时间范围

# 添加图表信息
plt.xlabel('Time (Adjusted)')
plt.ylabel('Value')
plt.title('Comparison of Time Error and PTP Master Offset')
plt.legend()
plt.grid()
#导出矢量图
plt.savefig(r".\0507\comparison_all_metrics.svg")
plt.savefig(r".\0507\comparison_all_metrics_adjusted.png")
plt.show()