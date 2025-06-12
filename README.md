<!-- filepath: c:\Git_Code\Clock_Sync\README.md -->
# 项目文件结构

本项目用于记录和分析不同时间同步方法的精度。文件结构组织如下：

```
Clock_Sync/
├── README.md  # 项目总览，说明文件结构、数据格式等
│
├── data/      # 存放原始测试数据
│   ├── phase_deviation_0524/      # 相位偏差相关实验数据 (原 0524 文件夹内容)
│   ├── temperature_effects_0607/  # 温度影响相关实验数据 (原 0607 文件夹部分内容)
│   └── load_tests_0612/           # 不同负载下的测试数据 (原 0612 文件夹内容)
│       ├── cpu_load/
│       └── traffic_load/
│
├── scripts/   # 存放分析脚本和模型代码
│   ├── analysis/                  # 数据分析和绘图脚本
│   │   └── performance_comparison.py  # (原 0524/performance_cmp.py)
│   │   └── ... (其他特定分析脚本可放于此)
│   └── models/                    # 独立模型或算法实现
│       └── kalman_filter.py         # (原 0607/kalman.py)
│
├── notebooks/ # Jupyter Notebooks 用于探索性数据分析和可视化
│   └── clock_drift_analysis.ipynb # (原 0607/drift.ipynb)
│
├── results/   # 存放由脚本生成的图表、报告等产出物
│   └── phase_deviation_0524/      # (原 0524 文件夹中的图表)
│   └── ... (其他实验结果的对应子文件夹)
│
└── archive/   # 归档旧的、格式不一致或不再积极使用的数据和分析
    ├── 0507/
    └── old_data_analysis/
```

--- 

## 数据格式说明

### `log0-original*.log` 文件格式 (例如 `log0-original(低温).log`)

这些日志文件记录了详细的时钟同步过程中的多维度数据。其主要格式如下：

```
// REAL 数据段（字段1-10）
 ns2s(getT1(REAL)), ns2s(getT2(REAL)),
 ns2s(getT3(REAL)), ns2s(getT4(REAL)),
 ns2us(getT2T1(REAL)), ns2us(getT3T4(REAL)),
 ns2us(getOffset(REAL)), ns2us(getCalculatedOffset(REAL)),
 ns2us(getDelay(REAL)), ns2s(getT4T1(REAL)),

// RAW时钟数据段（字段11-20）
 ns2s(getT1(RAW)), ns2s(getT2(RAW)),
 ns2s(getT3(RAW)), ns2s(getT4(RAW)),
 ns2us(getT2T1(RAW)), ns2us(getT3T4(RAW)),
 ns2us(getOffset(RAW)), ns2us(getCalculatedOffset(RAW)),
 ns2us(getDelay(RAW)), ns2s(getT4T1(RAW)),

// 元数据段（字段21-28）
 seqNo, // 序列号
 ns2us(t1RealKernel - t1RealApp), // T1 内核与应用层时间戳差异
 ns2us(t2RealApp - t2RealKernel), // T2 应用层与内核时间戳差异
 ns2us(t3RealKernel - t3RealApp), // T3 内核与应用层时间戳差异
 ns2us(t4RealApp - t4RealKernel), // T4 应用层与内核时间戳差异
 ns2us(models[RAW1.local.calculate0ffsetToRemote(getT1(RAW)))), // 字段 26: 基于RAW T1计算的到远端的偏移
 ns2us(getT2T1(RAW)-models[RAW1.local.calculate0ffsetToRemote(getT1(RAW)))), // 字段 27: RAW T2-T1 与字段26的差值
 ns2us(models[RAW1.local.calculate0ffsetToRemote(getT1(RAW))-getT3T4(RAW))) // 字段 28: 字段26与RAW T3-T4的差值
```

**字段说明:**
* `ns2s()`: 纳秒转换为秒的函数。
* `ns2us()`: 纳秒转换为微秒的函数。
* `getT1(REAL)`, `getT2(REAL)`, ...: 获取经过校准/真实时间戳的函数。
* `getT1(RAW)`, `getT2(RAW)`, ...: 获取原始/未校准时间戳的函数。
* `getOffset(REAL/RAW)`: 获取计算出的时间偏移。
* `getCalculatedOffset(REAL/RAW)`: 另一种计算出的时间偏移（可能经过滤波或其他处理）。
* `getDelay(REAL/RAW)`: 获取网络延迟。
* `tXRealKernel`, `tXRealApp`: 分别表示在内核层和应用层记录的时间戳。
* `models[RAW1.local.calculate0ffsetToRemote(...)]`: 基于模型计算的偏移量。

### `timeError*.log` 文件格式 (例如 `data/phase_deviation_0524/timeError.log`)

该文件记录时间戳及其对应的偏差值。

*   **第一列**: 时间戳 (单位：秒)。
*   **第二列**: 偏差值 (单位：可能为微秒或纳秒，具体需根据实验上下文确定)。

示例:
```
0.000 -165.387
0.115 -169.939
0.232 -191.993
...
```

### `cpu_temp*.log` 文件格式 (例如 `data/temperature_effects_0607/cpu_temp(低温).log`)

该文件记录时间或序列号对应的CPU温度。

*   **第一列**: 序列号或时间戳。
*   **第二列**: CPU温度 (单位：°C)。

示例:
```
0 49.60 °C
1 49.60 °C
3 49.60 °C
...
```
