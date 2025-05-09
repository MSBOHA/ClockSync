import os
import pandas as pd
import matplotlib.pyplot as plt

# 修正文件路径，使用原始字符串 r"" 避免转义问题
time_error_log_file_path = r".\0507\自有方法\timeError.log"

# 读取数据
data = pd.read_csv(time_error_log_file_path, sep=" ", header=None)
data.columns = ['time', 'error']

# 将 time 列转换为数值类型（假设 time 是秒数或其他数值格式）
data['time'] = pd.to_numeric(data['time'], errors='coerce')

# 筛选出中间时间段的数据（假设时间范围为 100 到 900）
filtered_data = data[(data['time'] > 5) & (data['time'] < 390)]

# 绘制筛选后的 error 随时间变化的图
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['time'], filtered_data['error'], label='Filtered Error', color='green')
plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Filtered Error over Time')
plt.legend()
plt.grid()
plt.savefig(r".\0507\自有方法\filtered_error_over_time.png")
plt.show()

# 计算筛选后 error 的统计特性
mean_error = filtered_data['error'].mean()
std_error = filtered_data['error'].std()
print(f"Filtered Mean Error: {mean_error}")
print(f"Filtered Standard Deviation of Error: {std_error}")