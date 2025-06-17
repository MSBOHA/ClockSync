import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_allan_variance(phi_series, m_values, tau_0):
    N = len(phi_series)
    phi_series = np.array(phi_series, dtype=float)
    allan_vars, actual_taus = [], []
    for m in m_values:
        if N <= 2 * m: break
        tau = m * tau_0
        diff = phi_series[2*m:] - 2*phi_series[m:-m] + phi_series[:-2*m]
        variance = np.sum(diff**2) / (2 * (N - 2*m) * tau**2)
        allan_vars.append(variance)
        actual_taus.append(tau)
    return np.array(actual_taus), np.array(allan_vars)

# 以项目根目录为基准，保证点击运行按钮也能找到数据
DATA_DIR = os.path.join('data', 'temp_drift_0613')
LOG_FILE = os.path.join(DATA_DIR, 'log0-original101.log')

# 严格按照README的28个字段名
column_names = [
    "ns2s(getT1(REAL))", "ns2s(getT2(REAL))", "ns2s(getT3(REAL))", "ns2s(getT4(REAL))",
    "ns2us(getT2T1(REAL))", "ns2us(getT3T4(REAL))", "ns2us(getOffset(REAL))", "ns2us(getCalculatedOffset(REAL))",
    "ns2us(getDelay(REAL))", "ns2s(getT4T1(REAL))",
    "ns2s(getT1(RAW))", "ns2s(getT2(RAW))", "ns2s(getT3(RAW))", "ns2s(getT4(RAW))",
    "ns2us(getT2T1(RAW))", "ns2us(getT3T4(RAW))", "ns2us(getOffset(RAW))", "ns2us(getCalculatedOffset(RAW))",
    "ns2us(getDelay(RAW))", "ns2s(getT4T1(RAW))",
    "seqNo", "ns2us(t1RealKernel - t1RealApp)", "ns2us(t2RealApp - t2RealKernel)", "ns2us(t3RealKernel - t3RealApp)",
    "ns2us(t4RealApp - t4RealKernel)", "ns2us(models[RAW1.local.calculate0ffsetToRemote(getT1(RAW))])",
    "ns2us(getT2T1(RAW)-models[RAW1.local.calculate0ffsetToRemote(getT1(RAW))])",
    "ns2us(models[RAW1.local.calculate0ffsetToRemote(getT1(RAW))-getT3T4(RAW)])"
]

def plot_allan_variance(log_file, output_dir='results', column='ns2s(getT1(RAW))', label='Allan 方差 (T1)', title='RAW T1 的 Allan 方差', m_start=2, m_end=None, m_step=1, max_points=None):
    df = pd.read_csv(log_file, sep='\s+', header=None)
    if len(df.columns) >= len(column_names):
        df.columns = column_names + [f"col_{i}" for i in range(len(df.columns)-len(column_names))]
    else:
        df.columns = column_names[:len(df.columns)]
    t1 = df[column].values
    # 下采样（如有需要）
    if max_points is not None and len(t1) > max_points:
        idx = np.linspace(0, len(t1)-1, max_points, dtype=int)
        t1 = t1[idx]
    tau_0 = np.mean(np.diff(t1)) if len(t1) > 1 else 1.0
    if m_end is None:
        m_end = max(10, (len(t1)-1)//2)
    m_values = range(m_start, m_end, m_step)
    taus, allan_vars = calculate_allan_variance(t1, m_values, tau_0)
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.loglog(taus, allan_vars, marker='o', linestyle='-', label=label)
    plt.xlabel('平均时间 τ (秒)', fontproperties='SimHei')
    plt.ylabel('Allan 方差 σ²y(τ)', fontproperties='SimHei')
    plt.title(title, fontproperties='SimHei')
    plt.grid(True, which="both", ls="-")
    plt.legend(prop={'family': 'SimHei'})
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'allan_variance.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()
    return taus, allan_vars

# 保留原main兼容命令行

def main():
    plot_allan_variance(LOG_FILE, output_dir=os.path.join('results', 'temp_drift_0613'), column='ns2s(getT1(RAW))', label='Allan 方差 (T1)', title='101号设备 RAW T1 的 Allan 方差')

if __name__ == '__main__':
    main()
