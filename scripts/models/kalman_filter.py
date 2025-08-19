import numpy as np
import matplotlib.pyplot as plt

# --- 1. 参数设置 (严格按照论文定义) ---
Delta_t = 10  # 同步周期 (s), 论文中使用 Delta_t 表示时间步长
num_steps = 20000 # 仿真步数

# 时钟噪声参数 (论文第 III 节和 IV 节会给出详细解释和典型值)
# 假设这里采用论文中提及的晶振时钟的典型参数
# 您需要根据论文图表或具体案例来选择这些值
# 以下为示例值，用于演示，并非论文中的确切值
# 通常 sigma_theta^2 远小于 sigma_gamma^2，且它们都非常小
sigma_theta_sq = 1e-12 # 相位噪声功率谱密度 (s^2/Hz)
sigma_gamma_sq = 1e-16 # 频率噪声功率谱密度 (1/Hz)


# 状态转移矩阵 A (论文式 9)
A = np.array([[1, Delta_t],
              [0, 1]])

# 过程噪声协方差矩阵 Q (论文式 11)
Q = np.array([[sigma_theta_sq * Delta_t, 0],
              [0, sigma_gamma_sq * Delta_t]])

# 观测矩阵 H (论文式 12)
H = np.eye(2) # 更新为 2x2 单位矩阵

# 测量噪声协方差 R (论文式 13/33)
# 根据提供的图片公式更新 R
# R = [[σ_0M^2, σ_θM^2/ΔT], [σ_θM^2/ΔT, 2*(σ_θM/ΔT)^2]]
# σ_0M^2 -> sigma_nu_sq (offset measurement variance)
# σ_θM^2 -> sigma_theta_sq (phase noise related term)
# ΔT -> Delta_t
R = np.array([
    [sigma_gamma_sq, sigma_theta_sq / Delta_t],
    [sigma_theta_sq / Delta_t, 2 * sigma_theta_sq / (Delta_t**2)]
])

# --- 2. 仿真时钟模型初始化 (真实时钟状态) ---
# 真实时钟初始偏移和频率偏移，用于模拟真实世界的时钟行为
true_x = np.array([[1],  # 真实时钟偏移 (s)
                   [5e-8]]) # 真实时钟频率偏移 (s/s) - 模拟一个初始频率偏差

# 存储真实值和测量值
true_offsets = []
measured_offsets = []
true_freq_offsets = []

# --- 3. 卡尔曼滤波器初始化 ---
# 初始状态估计 (通常设为0，或者根据先验知识)
kalman_x_hat = np.array([[0.0],
                         [0.0]])

# 初始误差协方差矩阵 P (表示对初始估计的不确定性)
# 可以设置为较大的值，让滤波器快速收敛
kalman_P = np.array([[1e-2, 0],
                     [0, 1e-6]]) # 1e-2 s^2 for offset, 1e-6 (s/s)^2 for freq offset

# 存储卡尔曼滤波器的估计值
kalman_estimated_offsets = []
kalman_estimated_freq_offsets = []
# ...existing code...

# OU噪声参数（可根据需要调整）
ou_theta = 0.0          # OU过程的均值
ou_tau = 1000           # 时间常数（秒），决定回归均值的快慢
ou_sigma = np.sqrt(sigma_gamma_sq)  # OU过程的强度

# 初始化OU过程的频率噪声分量
ou_freq_noise = 0.0

# --- 4. 仿真循环 ---
# for k in range(num_steps):
#     # a. 生成真实时钟行为
#     # 偏移项仍用高斯白噪声
#     phase_noise = np.random.normal(0, np.sqrt(Q[0,0]))
#     # 频率项用OU过程
#     ou_freq_noise += (ou_theta - ou_freq_noise) * (Delta_t / ou_tau) \
#                      + ou_sigma * np.sqrt(Delta_t) * np.random.normal(0, 1)
#     process_noise = np.array([[phase_noise],
#                               [ou_freq_noise]])
    
#     # 真实状态更新
#     true_x = np.dot(A, true_x) + process_noise
    # ...existing code...
# --- 4. 仿真循环 ---
for k in range(num_steps):
    # a. 生成真实时钟行为
    # 根据Q的对角线元素生成独立的、零均值的高斯过程噪声
    process_noise = np.array([[np.random.normal(0, np.sqrt(Q[0,0]))],
                              [np.random.normal(0, np.sqrt(Q[1,1]))]])
    
    # 真实状态更新
    true_x = np.dot(A, true_x) + process_noise
    true_offsets.append(true_x[0, 0])
    true_freq_offsets.append(true_x[1, 0])

    # b. 生成测量值 (在真实偏移和频率上叠加测量噪声)
    # H is eye(2), R is 2x2. Measurement z is [offset_measured, freq_offset_measured]
    measurement_noise_mean = [0, 0]
    measurement_noise_val = np.random.multivariate_normal(measurement_noise_mean, R).reshape(2, 1) # 测量噪声 (2x1 vector)
    # measured_z = H @ true_x + measurement_noise_val. Since H = eye(2), measured_z = true_x + noise
    measured_z = true_x + measurement_noise_val # 测量值 (2x1 vector)
    measured_offsets.append(measured_z[0, 0]) # 存储测量偏移用于绘图

    # --- 卡尔曼滤波器步骤 ---
    # 预测 (Prediction)
    kalman_x_hat_minus = np.dot(A, kalman_x_hat)
    kalman_P_minus = np.dot(np.dot(A, kalman_P), A.T) + Q

    # 更新 (Update)
    innovation_covariance = np.dot(np.dot(H, kalman_P_minus), H.T) + R
    K = np.dot(np.dot(kalman_P_minus, H.T), np.linalg.inv(innovation_covariance)) # 卡尔曼增益

    measurement_residual = measured_z - np.dot(H, kalman_x_hat_minus) # 测量残差 (新息)
    kalman_x_hat = kalman_x_hat_minus + np.dot(K, measurement_residual) # 更新状态估计
    kalman_P = np.dot((np.eye(2) - np.dot(K, H)), kalman_P_minus) # 更新误差协方差

    kalman_estimated_offsets.append(kalman_x_hat[0, 0])
    kalman_estimated_freq_offsets.append(kalman_x_hat[1, 0])

# --- 5. 结果可视化 ---
time_axis = np.arange(num_steps) * Delta_t

plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1)
plt.plot(time_axis, true_offsets, label='True Clock Offset', linewidth=2)
plt.plot(time_axis, measured_offsets, 'x', label='Measured Offset (Noisy)', alpha=0.3, markersize=4)
plt.plot(time_axis, kalman_estimated_offsets, label='Kalman Estimated Offset', linewidth=2, linestyle='--')
plt.title('Clock Offset Synchronization (Kalman Filter)')
plt.xlabel('Time (s)')
plt.ylabel('Offset (s)')
plt.legend()
plt.grid(True)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-9,-9)) # 科学计数法显示微秒级别

plt.subplot(2, 1, 2)
plt.plot(time_axis, true_freq_offsets, label='True Clock Frequency Offset', linewidth=2)
plt.plot(time_axis, kalman_estimated_freq_offsets, label='Kalman Estimated Frequency Offset', linewidth=2, linestyle='--')
plt.title('Clock Frequency Offset Estimation')
plt.xlabel('Time (s)')
plt.ylabel('Frequency Offset (s/s)')
plt.legend()
plt.grid(True)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-8,-8)) # 科学计数法显示

plt.tight_layout()
plt.show()

# 定量评估
rmse_kalman_offset = np.sqrt(np.mean((np.array(true_offsets) - np.array(kalman_estimated_offsets))**2))
rmse_measured_offset = np.sqrt(np.mean((np.array(true_offsets) - np.array(measured_offsets))**2))

print(f"RMSE for Measured Offset: {rmse_measured_offset:.2e} s")
print(f"RMSE for Kalman Estimated Offset: {rmse_kalman_offset:.2e} s")

# --- 6. 计算平均和最大偏差 ---
# 时间偏差 (偏移误差)
offset_errors = np.abs(np.array(true_offsets) - np.array(kalman_estimated_offsets))
avg_offset_error = np.mean(offset_errors)
max_offset_error = np.max(offset_errors)

# 频率偏差 (频率误差)
freq_errors = np.abs(np.array(true_freq_offsets) - np.array(kalman_estimated_freq_offsets))
avg_freq_error = np.mean(freq_errors)
max_freq_error = np.max(freq_errors)

print("\\n--- 额外统计 ---")
print(f"平均时间偏差 (Kalman 估计): {avg_offset_error:.2e} s")
print(f"最大时间偏差 (Kalman 估计): {max_offset_error:.2e} s")
print(f"平均频率偏差 (Kalman 估计): {avg_freq_error:.2e} s/s")
print(f"最大频率偏差 (Kalman 估计): {max_freq_error:.2e} s/s")