import numpy as np
import matplotlib.pyplot as plt
import allantools

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# --- 仿真参数 ---
N_points = 2**20  # 数据点数量
dt = 0.1         # 采样间隔 (秒)
rate = 1.0/dt     # 采样率 (Hz)

# 相位噪声 (白相位调制 - White PM)
# 直接模拟时间误差 x(t)
phase_noise_time_stddev = 1e-6  # 时间误差标准差：1 微秒 (µs)

# 频率噪声 (奥恩斯坦-乌伦贝克过程 - Ornstein-Uhlenbeck FM)
frac_freq_ou_stddev = 0.1e-6    # 分数频率标准差 (y): 0.1 ppm (百万分之一)
tau_correlation_ou = 100.0      # OU 噪声的相关时间 (秒) (例如 100秒)

# 频率漂移 (随机游走频率调制 - Random Walk FM)
# 这是分数频率 y(t) 的随机游走分量的斜率强度
# 单位: ppm / sqrt(秒) 或 1 / sqrt(秒) (取决于 frac_freq_ou_stddev 的单位约定)
# 这里我们假设其单位与 frac_freq_ou_stddev 的单位一致 (即 ppm/sqrt(s) 需要乘以 1e-6 得到绝对值)
# 或者直接给定绝对值，如 1e-9 /sqrt(s)
frac_freq_rw_slope_strength = 1e-9 # 例如: 0.001 ppm/sqrt(s) -> 1e-9 s^-0.5

# --- 生成噪声分量 ---

# 1. 高斯白相位噪声 (直接模拟时间误差 x_PM(t))
# 这代表 x_PM(t)
x_pm = np.random.normal(loc=0.0, scale=phase_noise_time_stddev, size=N_points)

# 2. 奥恩斯坦-乌伦贝克 (OU) 分数频率噪声 (手动实现)
# 这代表 y_FM_OU(t)
# 离散时间更新: y[k] = y[k-1] * exp(-b1*dt) + sigma_driving * N(0,1)
# 其中 b1 = 1/tau_correlation_ou
# 且 sigma_driving = frac_freq_ou_stddev * sqrt(1 - exp(-2*b1*dt))

b1_ou = 1.0 / tau_correlation_ou
y_fm_ou = np.zeros(N_points)

if N_points > 0:
    # 初始值来自平稳分布 N(0, frac_freq_ou_stddev)
    y_fm_ou[0] = np.random.normal(loc=0.0, scale=frac_freq_ou_stddev)

    # 预计算循环中使用的常量
    exp_term_ou = np.exp(-b1_ou * dt)
    innovation_std_ou = frac_freq_ou_stddev * np.sqrt(1 - np.exp(-2 * b1_ou * dt))

    # 生成噪声样本
    for i in range(1, N_points):
        white_noise_sample = np.random.normal(loc=0.0, scale=1.0)
        y_fm_ou[i] = y_fm_ou[i-1] * exp_term_ou + innovation_std_ou * white_noise_sample

# 积分 OU 分数频率噪声以获得其对相位误差(时间误差)的贡献
# x_FM_OU(t_k) = sum_{i=0}^{k-1} y_FM_OU(t_i) * dt
x_fm_ou_contrib = np.cumsum(y_fm_ou) * dt

# 3. 随机游走频率调制 (RWFM)
# 这代表 y_FM_RW(t)，其积分为对相位误差的贡献
# y_FM_RW[k] = y_FM_RW[k-1] + N(0, (frac_freq_rw_slope_strength * sqrt(dt))^2)
y_fm_rw = np.zeros(N_points)
if N_points > 0:
    # 初始频率漂移可以为零，或者如果存在先验知识，则从某个分布中抽取
    y_fm_rw[0] = 0.0 # 假设初始频率漂移为0
    
    # 随机游走每步的标准差
    rw_increment_std = frac_freq_rw_slope_strength * np.sqrt(dt)
    for i in range(1, N_points):
        white_noise_increment = np.random.normal(loc=0.0, scale=rw_increment_std)
        y_fm_rw[i] = y_fm_rw[i-1] + white_noise_increment

# 积分 RWFM 分数频率噪声以获得其对相位误差(时间误差)的贡献
x_fm_rw_contrib = np.cumsum(y_fm_rw) * dt


# --- 合并噪声分量得到总时间误差 ---
# 时钟1: WPM + 积分后的OU频率噪声 (y_fm_ou)
# 不再包含 RWFM 成分
x_total_time_error = x_pm + x_fm_ou_contrib

# --- 为第二个时钟（仅 WPM 和 RWFM）合并噪声分量 ---
# 时钟2: WPM + 积分后的RWFM频率噪声 (y_fm_rw)
# 时钟2只包含相位噪声 (x_pm) 和随机游走频率噪声引起的相位误差 (x_fm_rw_contrib)
x_total_time_error_clk2 = x_pm + x_fm_rw_contrib

# --- 计算艾伦方差 ---
min_tau = dt # 最小平均时间
max_tau = (N_points * dt) / 3  # tau 的典型上限
taus = np.logspace(np.log10(min_tau), np.log10(max_tau), num=50) # 对数间隔的平均时间

# 计算重叠艾伦方差 (oadev)
# 输入 data_type="phase" 表示输入是相位数据 (在我们的例子中是时间误差 x(t))
# rate 是测量速率 (1/dt)
(taus_out, adev, adeverr, adevn) = allantools.oadev(
    data=x_total_time_error,
    rate=rate,
    data_type="phase", # 输入数据类型为相位（时间误差）
    taus=taus
)

# --- 计算第二个时钟的艾伦方差 ---
(taus_out_clk2, adev_clk2, adeverr_clk2, adevn_clk2) = allantools.oadev(
    data=x_total_time_error_clk2,
    rate=rate,
    data_type="phase", # 输入数据类型为相位（时间误差）
    taus=taus # 使用与第一个时钟相同的 tau 点进行比较
)

# --- 绘制艾伦方差 ---
plt.figure(figsize=(12, 7)) # 调整图形大小以便容纳更长的标题
plt.loglog(taus_out, adev, marker='o', linestyle='-', label='时钟1 (WPM+OUFM)') # 更新第一个时钟的标签

# 绘制第二个时钟的艾伦方差
plt.loglog(taus_out_clk2, adev_clk2, marker='s', linestyle=':', label='时钟2 (WPM+RWFM)') # 使用不同标记和线型

# 添加常见噪声类型的参考斜率线
if len(taus_out) > 1 and len(adev) > 0:
    # 白相位调制 (WPM, ADEV ~ tau^-1)
    # 固定在 ADEV 图的起始附近，WPM 可能在此处占主导
    c_wpm = adev[0] * taus_out[0]**1.0
    plt.loglog(taus_out, c_wpm * taus_out**(-1.0), linestyle='--', color='grey', label=r'白相位调制 (WPM) ($\propto \tau^{-1}$)')

    # 白频率调制 (WFM, ADEV ~ tau^-0.5)
    # 当 tau << tau_correlation_ou 时，OU 噪声表现得像白频率调制。
    # 固定在合适的区域，例如 tau 较小但大于 dt 的地方。
    # 如果可能，找到 tau 在 tau_correlation_ou / 10 附近的索引
    idx_wfm_anchor = np.argmin(np.abs(taus_out - max(dt*5, tau_correlation_ou / 20.0)))
    if idx_wfm_anchor >= len(adev): # 处理 taus_out 较短的情况
        idx_wfm_anchor = len(adev) // 4 # 取一个靠前的点

    c_wfm = adev[idx_wfm_anchor] * taus_out[idx_wfm_anchor]**0.5
    plt.loglog(taus_out, c_wfm * taus_out**(-0.5), linestyle=':', color='blue', label=r'白频率调制 (WFM) ($\propto \tau^{-1/2}$)')

    # 随机游走频率调制 (RWFM, ADEV ~ tau^+0.5)
    # 当 tau >> tau_correlation_ou 时，OU 噪声可能也显示出 tau^+0.5 的行为，但这里我们更关注由 y_fm_rw 引入的 RWFM。
    # 固定在 ADEV 图的末尾附近，如果 RWFM 在此处预期出现。
    idx_rwfm_anchor = np.argmin(np.abs(taus_out - tau_correlation_ou * 5.0)) # 尝试一个锚点
    if idx_rwfm_anchor >= len(adev) or taus_out[idx_rwfm_anchor] < tau_correlation_ou * 2: # 确保锚点在预期区域内
         idx_rwfm_anchor = -1 # 如果特定锚点不合适，则使用最后一个点

    c_rwfm = adev[idx_rwfm_anchor] * taus_out[idx_rwfm_anchor]**(-0.5) # RWFM 斜率为 +0.5，所以这里应该是 adev = C * tau^0.5 => C = adev * tau^-0.5
    plt.loglog(taus_out, c_rwfm * taus_out**(0.5), linestyle='-.', color='green', label=r'随机游走频率调制 (RWFM) ($\propto \tau^{+1/2}$)')

plt.xlabel(r'平均时间, $\tau$ (秒)')
plt.ylabel(r'艾伦方差, $\sigma_y(\tau)$')
# 更新标题以包含所有噪声参数
title_str = (f'模拟时钟数据的艾伦方差\\n'
             f'($\\sigma_{{x_{{PM}}}}$={phase_noise_time_stddev*1e6:.1f}$\\mu\mathrm{{s}}$, '
             f'$\\sigma_{{y_{{OU}}}}$={frac_freq_ou_stddev*1e6:.1f}ppm, $\\tau_c$={tau_correlation_ou:.0f}s, '
             f'Slope$_{{RWFM}}$={frac_freq_rw_slope_strength*1e6:.3f}ppm/$\\sqrt{{s}}$)')
plt.title(title_str)
plt.grid(True, which="both", ls="-", alpha=0.5) # 添加网格线
plt.legend() # 显示图例
plt.tight_layout() # 自动调整子图参数，使之填充整个图像区域
plt.show() # 显示图形
