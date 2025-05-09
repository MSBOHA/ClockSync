import os
import pandas as pd
import matplotlib.pyplot as plt

# 定义文件路径
time_error_log_file_path_own = r".\0507\自有方法\timeError.log"
time_error_log_file_path_chrony = r".\0507\chrony\timeError.log"

# 定义一个函数来读取和处理数据
def process_time_error_log(file_path, label, color, output_image_path, time_range=None):
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

        # 绘制筛选后的 error 随时间变化的图
        plt.plot(filtered_data['time'], filtered_data['error'], label=label, color=color)

        # 计算筛选后 error 的统计特性
        mean_error = filtered_data['error'].mean()
        std_error = filtered_data['error'].std()
        print(f"{label} - Filtered Mean Error: {mean_error}")
        print(f"{label} - Filtered Standard Deviation of Error: {std_error}")
    else:
        print(f"文件不存在: {file_path}")

# 创建图表
plt.figure(figsize=(10, 6))

# 处理自有方法的 timeError.log
process_time_error_log(
    time_error_log_file_path_own,
    label="Own Method",
    color="green",
    output_image_path=r".\0507\自有方法\filtered_error_over_time.png",
    time_range=(5, 390)  # 可选的时间范围
)

# 处理 chrony 的 timeError.log
process_time_error_log(
    time_error_log_file_path_chrony,
    label="Chrony Method",
    color="blue",
    output_image_path=r".\0507\chrony\filtered_error_over_time.png",
    time_range=None  # 可选的时间范围
)

# 添加图表信息
plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Filtered Error over Time (Comparison)')
plt.legend()
plt.grid()

# 保存图表
plt.savefig(r".\0507\comparison_filtered_error_over_time.png")
plt.show()