# 温度与时钟频率关系分析
# 分析设备100和101的温度与其时钟频率漂移之间的关系
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
import os

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据加载

def load_sync_data(filepath, window_seconds=100):
    """加载同步数据并用滑窗斜率法计算频率漂移（使用RealTime 11-14列）"""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('//')]
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 14:
            try:
                t1, t2, t3, t4 = map(float, parts[10:14])  # 使用11-14列RealTime
                data.append([t1, t2, t3, t4])
            except ValueError:
                continue
    
    df = pd.DataFrame(data, columns=['t1', 't2', 't3', 't4'])
    df['timestamp'] = df['t1']
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 滑窗斜率法计算频率比
    time_span = df['timestamp'].max() - df['timestamp'].min()
    points_per_second = len(df) / time_span
    window_size = int(window_seconds * points_per_second)
    if window_size < 3:
        window_size = 3
    
    freq_ratio = np.full(len(df), np.nan)
    for i in range(len(df)):
        left = max(0, i - window_size//2)
        right = min(len(df), i + window_size//2 + 1)
        idx = np.arange(left, right)
        if len(idx) < 3:
            continue
        t1s = df['t1'].values[idx]
        t2s = df['t2'].values[idx]
        timestamps = df['timestamp'].values[idx]
        # 计算t1和t2相对于真实时间的斜率（即频率）
        k1 = np.polyfit(timestamps, t1s, 1)[0]  # dt1/dt_real
        k2 = np.polyfit(timestamps, t2s, 1)[0]  # dt2/dt_real
        if k1 != 0:
            freq_ratio[i] = k2 / k1  # 频率比
    
    df['freq_drift'] = freq_ratio
    return df[['timestamp', 't1', 't2', 't3', 't4', 'freq_drift']].dropna()

def load_temperature_data(filepath):
    """加载温度数据"""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    data = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                time_idx = float(parts[0])
                temp = float(parts[1])
                data.append([time_idx, temp])
            except ValueError:
                continue
    df = pd.DataFrame(data, columns=['time_idx', 'temperature'])
    return df

def interpolate_temperature(sync_ts, temp_data):
    """为同步时间戳插值温度数据"""
    return np.interp(sync_ts, temp_data['timestamp'], temp_data['temperature'])

# 联合模型定义

def joint_linear_ratio_model(temp_data, f01, a1, T01, f02, a2, T02):
    """联合线性模型：f1/f2 = [f01*(1+a1*(T1-T01))] / [f02*(1+a2*(T2-T02))]"""
    T1, T2 = temp_data
    f1 = f01 * (1 + a1 * (T1 - T01))
    f2 = f02 * (1 + a2 * (T2 - T02))
    return f1 / f2

def joint_quadratic_ratio_model(temp_data, f01, b1, T01, f02, b2, T02):
    """联合二次模型：f1/f2 = [f01*(1-b1*(T1-T01)^2)] / [f02*(1-b2*(T2-T02)^2)]"""
    T1, T2 = temp_data
    f1 = f01 * (1 - b1 * (T1 - T01)**2)
    f2 = f02 * (1 - b2 * (T2 - T02)**2)
    return f1 / f2

def fit_joint_models(temp1, temp2, freq):
    """拟合联合线性和二次模型（改进的优化算法）"""
    from scipy.optimize import differential_evolution, minimize
    
    T01_mean = np.mean(temp1)
    T02_mean = np.mean(temp2)
    T1_range = temp1.max() - temp1.min()
    T2_range = temp2.max() - temp2.min()
    freq_mean = np.mean(freq)
    
    print(f"温度范围: T1={temp1.min():.1f}-{temp1.max():.1f}°C, T2={temp2.min():.1f}-{temp2.max():.1f}°C")
    
    # === 线性模型拟合 ===
    def linear_objective(params):
        try:
            pred = joint_linear_ratio_model([temp1, temp2], *params)
            return np.mean((freq - pred)**2)
        except:
            return 1e10
    
    # 线性模型参数边界：[f01, a1, T01, f02, a2, T02]
    linear_bounds = [
        (0.99, 1.01),           # f01: 接近1
        (-1e-5, 1e-5),          # a1: 温度系数
        (temp1.min()-5, temp1.max()+5),  # T01: 在合理范围内
        (0.99, 1.01),           # f02: 接近1
        (-1e-5, 1e-5),          # a2: 温度系数
        (temp2.min()-5, temp2.max()+5)   # T02: 在合理范围内
    ]
    
    # 多次尝试线性拟合
    best_linear_result = None
    best_linear_rmse = np.inf
    
    for seed in range(5):
        try:
            np.random.seed(seed)
            result = differential_evolution(linear_objective, linear_bounds, seed=seed, maxiter=1000)
            if result.success and result.fun < best_linear_rmse:
                best_linear_result = result
                best_linear_rmse = result.fun
        except:
            continue
    
    if best_linear_result is None:
        # 回退到原方法
        p0_lin = [1.0, 1e-6, T01_mean, 1.0, 1e-6, T02_mean]
        popt_lin, _ = curve_fit(joint_linear_ratio_model, [temp1, temp2], freq, p0=p0_lin, maxfev=10000)
    else:
        popt_lin = best_linear_result.x
    
    pred_lin = joint_linear_ratio_model([temp1, temp2], *popt_lin)
    rmse_lin = np.sqrt(np.mean((freq - pred_lin)**2))
    
    # === 二次模型拟合 ===
    def quad_objective(params):
        try:
            pred = joint_quadratic_ratio_model([temp1, temp2], *params)
            if np.any(~np.isfinite(pred)):
                return 1e10
            return np.mean((freq - pred)**2)
        except:
            return 1e10
    
    # 二次模型参数边界：[f01, b1, T01, f02, b2, T02]
    quad_bounds = [
        (0.95, 1.05),           # f01: 更宽松的基准频率范围
        (-1e-6, 1e-6),          # b1: 二次系数
        (temp1.min()-2, temp1.max()+2),  # T01: 参考温度在观测范围内
        (0.95, 1.05),           # f02: 更宽松的基准频率范围
        (-1e-6, 1e-6),          # b2: 二次系数
        (temp2.min()-2, temp2.max()+2)   # T02: 参考温度在观测范围内
    ]
    
    # 多次尝试二次拟合
    best_quad_result = None
    best_quad_rmse = np.inf
    
    for seed in range(10):  # 二次模型更难拟合，多试几次
        try:
            np.random.seed(seed)
            result = differential_evolution(quad_objective, quad_bounds, seed=seed, maxiter=1500)
            if result.success and result.fun < best_quad_rmse:
                best_quad_result = result
                best_quad_rmse = result.fun
        except:
            continue
    
    if best_quad_result is None:
        # 回退到原方法
        p0_quad = [1.0, 1e-7, T01_mean, 1.0, 1e-7, T02_mean]
        try:
            popt_quad, _ = curve_fit(joint_quadratic_ratio_model, [temp1, temp2], freq, 
                                   p0=p0_quad, maxfev=20000,
                                   bounds=([0.95, -1e-6, temp1.min()-2, 0.95, -1e-6, temp2.min()-2],
                                          [1.05, 1e-6, temp1.max()+2, 1.05, 1e-6, temp2.max()+2]))
        except:
            popt_quad = p0_quad
    else:
        popt_quad = best_quad_result.x
    
    pred_quad = joint_quadratic_ratio_model([temp1, temp2], *popt_quad)
    rmse_quad = np.sqrt(np.mean((freq - pred_quad)**2))
    
    return popt_lin, pred_lin, rmse_lin, popt_quad, pred_quad, rmse_quad

def plot_surface_with_obs(temp1, temp2, freq, model, params, title, sample_n=1000):
    """绘制3D曲面并叠加观测点"""
    T1 = np.linspace(temp1.min(), temp1.max(), 40)
    T2 = np.linspace(temp2.min(), temp2.max(), 40)
    T1g, T2g = np.meshgrid(T1, T2)
    
    if model == 'linear':
        Z = joint_linear_ratio_model([T1g, T2g], *params)
    else:
        Z = joint_quadratic_ratio_model([T1g, T2g], *params)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制曲面
    surf = ax.plot_surface(T1g, T2g, Z, cmap='viridis', alpha=0.8)
    
    # 采样观测点
    n = len(temp1)
    if n > sample_n:
        idx = np.random.choice(n, sample_n, replace=False)
        ax.scatter(temp1[idx], temp2[idx], freq[idx], 
                  c='red', s=15, alpha=0.8, label=f'观测点(采样{sample_n}个)')
    else:
        ax.scatter(temp1, temp2, freq, c='red', s=15, alpha=0.8, label='观测点')
    
    ax.set_xlabel('设备100温度 (°C)')
    ax.set_ylabel('设备101温度 (°C)')
    ax.set_zlabel('频率比值 (f1/f2)')
    ax.set_title(title)
    ax.legend()
    plt.colorbar(surf, shrink=0.5, aspect=20)
    plt.tight_layout()
    plt.show()

def plot_T101_vs_freq_fixed_T100(temp1, temp2, freq, popt_lin, popt_quad, T100_target=36.0, delta=1.0):
    """绘制固定设备100温度时，设备101温度与频率比的关系"""
    # 筛选T100在目标温度附近的数据点
    mask = np.abs(temp1 - T100_target) <= delta
    temp2_sel = temp2[mask]
    freq_sel = freq[mask]
    
    if len(temp2_sel) < 10:
        print(f"设备100温度{T100_target}°C附近的数据点太少: {len(temp2_sel)}个")
        return
    
    # 创建预测曲线的温度范围
    T101_range = np.linspace(temp2_sel.min(), temp2_sel.max(), 100)
    T100_fixed = np.full_like(T101_range, T100_target)
    
    # 线性模型预测
    pred_linear = joint_linear_ratio_model([T100_fixed, T101_range], *popt_lin)
    # 二次模型预测
    pred_quad = joint_quadratic_ratio_model([T100_fixed, T101_range], *popt_quad)
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(temp2_sel, freq_sel, c='blue', s=20, alpha=0.7, label=f'观测点({len(temp2_sel)}个)')
    plt.plot(T101_range, pred_linear, 'r-', linewidth=2, label='线性模型预测')
    plt.plot(T101_range, pred_quad, 'g-', linewidth=2, label='二次模型预测')
    
    plt.xlabel('设备101温度 (°C)')
    plt.ylabel('频率比值 (f1/f2)')
    plt.title(f'固定设备100温度={T100_target:.1f}°C时，设备101温度-频率比关系\n(温度范围: {T100_target-delta:.1f}°C ~ {T100_target+delta:.1f}°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"固定T100={T100_target:.1f}°C分析:")
    print(f"  筛选数据点: {len(temp2_sel)}个")
    print(f"  T101温度范围: {temp2_sel.min():.1f}°C ~ {temp2_sel.max():.1f}°C")
    print(f"  频率比范围: {freq_sel.min():.6f} ~ {freq_sel.max():.6f}")

def main():
    # 数据加载
    print("正在加载数据...")
    sync_data = load_sync_data('data/temp_drift_0613/log0-original101.log', window_seconds=100)
    temp_100 = load_temperature_data('data/temp_drift_0613/cpu_temp100.log')
    temp_101 = load_temperature_data('data/temp_drift_0613/cpu_temp101.log')
    
    # 时间对齐
    sync_start_time = sync_data['timestamp'].min()
    temp_100['timestamp'] = temp_100['time_idx'] + sync_start_time
    temp_101['timestamp'] = temp_101['time_idx'] + sync_start_time
    sync_data['temp_100'] = interpolate_temperature(sync_data['timestamp'], temp_100)
    sync_data['temp_101'] = interpolate_temperature(sync_data['timestamp'], temp_101)
    
    # 数据清洗
    q1 = sync_data['freq_drift'].quantile(0.01)
    q99 = sync_data['freq_drift'].quantile(0.99)
    clean_data = sync_data[(sync_data['freq_drift'] >= q1) & (sync_data['freq_drift'] <= q99)].copy()
    
    print(f"数据概况：")
    print(f"  总数据点: {len(sync_data):,}")
    print(f"  清洗后数据点: {len(clean_data):,}")
    print(f"  频率比范围: [{clean_data['freq_drift'].min():.6f}, {clean_data['freq_drift'].max():.6f}]")
    print(f"  设备100温度范围: {clean_data['temp_100'].min():.1f}°C - {clean_data['temp_100'].max():.1f}°C")
    print(f"  设备101温度范围: {clean_data['temp_101'].min():.1f}°C - {clean_data['temp_101'].max():.1f}°C")
    
    # 联合模型拟合
    print("\n正在拟合联合模型...")
    popt_lin, pred_lin, rmse_lin, popt_quad, pred_quad, rmse_quad = fit_joint_models(
        clean_data['temp_100'], clean_data['temp_101'], clean_data['freq_drift'])
    
    # 输出结果
    print("\n=== 拟合结果 ===")
    print(f"线性模型: f1/f2 = [f01*(1+a1*(T1-T01))] / [f02*(1+a2*(T2-T02))]")
    print(f"  参数: f01={popt_lin[0]:.6f}, a1={popt_lin[1]:.2e}, T01={popt_lin[2]:.2f}")
    print(f"        f02={popt_lin[3]:.6f}, a2={popt_lin[4]:.2e}, T02={popt_lin[5]:.2f}")
    print(f"  RMSE: {rmse_lin:.6e}")
    
    print(f"\n二次模型: f1/f2 = [f01*(1-b1*(T1-T01)^2)] / [f02*(1-b2*(T2-T02)^2)]")
    print(f"  参数: f01={popt_quad[0]:.6f}, b1={popt_quad[1]:.2e}, T01={popt_quad[2]:.2f}")
    print(f"        f02={popt_quad[3]:.6f}, b2={popt_quad[4]:.2e}, T02={popt_quad[5]:.2f}")
    print(f"  RMSE: {rmse_quad:.6e}")
    
    print(f"\n模型比较: {'二次模型更优' if rmse_quad < rmse_lin else '线性模型更优'} (RMSE差值: {abs(rmse_lin-rmse_quad):.6e})")
      # 绘制3D曲面
    print("\n正在绘制3D曲面...")
    plot_surface_with_obs(clean_data['temp_100'].values, clean_data['temp_101'].values, 
                         clean_data['freq_drift'].values, 'linear', popt_lin, 
                         f'联合线性模型频率比曲面 (RMSE: {rmse_lin:.6e})')
    
    plot_surface_with_obs(clean_data['temp_100'].values, clean_data['temp_101'].values,
                         clean_data['freq_drift'].values, 'quadratic', popt_quad,
                         f'联合二次模型频率比曲面 (RMSE: {rmse_quad:.6e})')
    
    # 额外分析：固定T100=36°C时的T101-频率比关系
    print("\n正在分析固定T100=36°C时的情况...")
    plot_T101_vs_freq_fixed_T100(clean_data['temp_100'].values, clean_data['temp_101'].values,
                                 clean_data['freq_drift'].values, popt_lin, popt_quad, T100_target=36.0)

if __name__ == "__main__":
    main()
