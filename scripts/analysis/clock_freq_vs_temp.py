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
    temp_df = pd.read_csv(TEMP_FILE, sep='\s+', header=None, names=['index', 'temperature', 'unit'])
    # 去除温度单位列，确保温度为float
    temp_df['temperature'] = temp_df['temperature'].astype(float)
    return df, temp_df

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
    df, temp_df = read_data()
    t1 = df["ns2s(getT1(RAW))"].values
    t2 = df["ns2s(getT2(RAW))"].values
    # 使用滑动窗口，step=10，segment_length可调
    centers, slopes = segment_slope_fit(t1, t2, segment_length=400, step=100)

    plt.figure(figsize=(12, 6))
    plt.plot(centers, slopes, marker='o', linestyle='-', label='分段频率比 (T1 vs T2)')
    plt.xlabel('T1 (秒)', fontproperties='SimHei')
    plt.ylabel('频率比 (斜率)', fontproperties='SimHei')
    plt.title('T1-T2 分段频率漂移', fontproperties='SimHei')
    plt.grid()
    plt.legend(prop={'family': 'SimHei'})
    plt.tight_layout()
    plt.show()

    # 温度曲线
    plt.figure(figsize=(12, 6))
    plt.plot(temp_df['index'], temp_df['temperature'], 'o-', label='CPU温度', markersize=3)
    plt.title('101号设备 CPU温度变化', fontproperties='SimHei')
    plt.xlabel('序号', fontproperties='SimHei')
    plt.ylabel('温度 (°C)', fontproperties='SimHei')
    plt.grid()
    plt.legend(prop={'family': 'SimHei'})
    plt.tight_layout()
    plt.show()

    # 可选：频率比与温度对比
    if len(centers) == len(temp_df['temperature']):
        plt.figure(figsize=(12, 6))
        plt.plot(centers, slopes, label='频率比 (T1 vs T2)')
        plt.plot(centers, temp_df['temperature'][:len(centers)], label='温度 (°C)')
        plt.xlabel('T1 (秒)', fontproperties='SimHei')
        plt.title('频率比与温度对比 (101号设备)', fontproperties='SimHei')
        plt.legend(prop={'family': 'SimHei'})
        plt.grid()
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()
