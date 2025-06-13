import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 以项目根目录为基准，保证点击运行按钮也能找到数据
DATA_DIR = os.path.join('data', 'temp_drift_0613')
LOG_FILE = os.path.join(DATA_DIR, 'log0-original101.log')
TEMP_FILE = os.path.join(DATA_DIR, 'cpu_temp101.log')

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

def read_data():
    df = pd.read_csv(LOG_FILE, sep='\s+', header=None)
    if len(df.columns) >= len(column_names):
        df.columns = column_names + [f"col_{i}" for i in range(len(df.columns)-len(column_names))]
    else:
        df.columns = column_names[:len(df.columns)]
    temp101 = pd.read_csv(TEMP_FILE, sep='\s+', header=None, names=['index', 'temperature', 'unit'], encoding='gbk')
    temp101['temperature'] = temp101['temperature'].astype(float)
    # 读取100号设备温度
    temp100_file = os.path.join(DATA_DIR, 'cpu_temp100.log')
    temp100 = pd.read_csv(temp100_file, sep='\s+', header=None, names=['index', 'temperature', 'unit'], encoding='gbk')
    temp100['temperature'] = temp100['temperature'].astype(float)
    return df, temp101, temp100

def segment_slope_fit(x, y, segment_length=150, step=1000):
    """滑动窗口分段拟合，step可调，默认1更平滑"""
    slopes, centers = [], []
    for start in range(0, len(x) - segment_length + 1, step):
        end = start + segment_length
        if end > len(x): break
        x_segment = x[start:end]
        y_segment = y[start:end]
        if len(x_segment) < 2: continue
        slope, _ = np.polyfit(x_segment, y_segment, 1)
        slopes.append(slope)
        centers.append(np.mean(x_segment))
    return np.array(centers), np.array(slopes)

def main():
    df, temp101, temp100 = read_data()
    t1 = df["ns2s(getT1(RAW))"].values
    t2 = df["ns2s(getT2(RAW))"].values
    # 使用滑动窗口，step=10，segment_length可调
    segment_length = 400
    step = 100
    centers, slopes = segment_slope_fit(t1, t2, segment_length=segment_length, step=step)
    window_indices = np.arange(len(centers))

    # 输出图片到 results/temp_drift_0613/
    output_dir = os.path.join('results', 'temp_drift_0613')
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(window_indices, slopes, marker='o', linestyle='-', label='分段频率比 (T1 vs T2)')
    plt.xlabel('滑动窗口序号', fontproperties='SimHei')
    plt.ylabel('频率比 (斜率)', fontproperties='SimHei')
    plt.title('T1-T2 分段频率漂移', fontproperties='SimHei')
    plt.grid()
    plt.legend(prop={'family': 'SimHei'})
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'freq_segment_vs_temp.png'))
    plt.show()

    # 温度曲线（100和101设备）
    plt.figure(figsize=(12, 6))
    plt.plot(temp101['index'], temp101['temperature'], 'o-', label='101号CPU温度', markersize=3)
    plt.plot(temp100['index'], temp100['temperature'], 'o-', label='100号CPU温度', markersize=3)
    plt.title('100/101号设备 CPU温度变化', fontproperties='SimHei')
    plt.xlabel('序号', fontproperties='SimHei')
    plt.ylabel('温度 (°C)', fontproperties='SimHei')
    plt.grid()
    plt.legend(prop={'family': 'SimHei'})
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpu_temp_curve.png'))
    plt.show()

    # 可选：频率比与温度对比（101号）
    if len(window_indices) == len(temp101['temperature']):
        plt.figure(figsize=(12, 6))
        plt.plot(window_indices, slopes, label='频率比 (T1 vs T2)')
        plt.plot(window_indices, temp101['temperature'][:len(window_indices)], label='101号温度 (°C)')
        plt.xlabel('滑动窗口序号', fontproperties='SimHei')
        plt.title('频率比与温度对比 (101号设备)', fontproperties='SimHei')
        plt.legend(prop={'family': 'SimHei'})
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'freq_vs_temp_compare.png'))
        plt.show()

if __name__ == '__main__':
    main()
