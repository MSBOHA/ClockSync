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
│   │   ├── delayAnalyzer/         # 延迟分析器 - 负载影响建模与分析
│   │   │   ├── data_processor.py    # 数据处理器：GMM拟合、分段分析、统计计算
│   │   │   └── load_analysis.py     # 负载分析：批量处理不同负载下的延迟分布
│   │   ├── performance_comparison.py  # (原 0524/performance_cmp.py)
│   │   └── ... (其他特定分析脚本可放于此)
│   ├── simulator/                 # 日志模拟器 - 温度和负载影响模拟
│   │   ├── log_processor.py         # 日志处理器：应用温度/负载效应到原始日志
│   │   └── simple_comparison.py     # 对比分析：原始vs处理后日志的分布对比
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

## 主要工具功能说明

### 延迟分析器 (delayAnalyzer)

**位置**: `scripts/analysis/delayAnalyzer/`

#### 1. 数据处理器 (`data_processor.py`)
负责处理时钟同步日志，提取延迟数据并进行统计建模：

**核心功能**:
- **延迟提取**: 从RAW时间戳(t1,t2,t3,t4)计算网络延迟
- **GMM建模**: 使用高斯混合模型拟合延迟分布
- **分段分析**: 支持按负载区间分段拟合，提高模型精度
- **统计分析**: 计算延迟分布的各种统计量(均值、标准差、分位数等)
- **可视化**: 生成延迟分布图、ECDF图、GMM拟合效果图

**支持的延迟类型**:
- 自估上行延迟: `(t2-t1)/2`
- 自估下行延迟: `(t4-t3)/2`  
- 平均延迟: `((t4-t1)-(t3-t2))/2`

**输出**:
- GMM参数文件 (.pkl, .json)
- 统计报告 (.csv, .txt)
- 可视化图表 (.png)

#### 2. 负载分析器 (`load_analysis.py`)
批量分析不同负载条件下的延迟特征：

**功能**:
- 批量处理多个负载等级的日志文件
- 生成负载vs延迟的对比报告
- 支持CPU负载和网络流量负载分析
- 自动化GMM拟合和统计分析流程

### 日志模拟器 (simulator)

**位置**: `scripts/simulator/`

#### 1. 日志处理器 (`log_processor.py`)
基于温度-频率模型和负载影响模型，对原始日志施加噪声：

**核心功能**:
- **温度效应建模**: 基于二次函数模型 `f = f0*(1-b*(T-T0)^2)` 模拟温度对时钟频率的影响
- **负载影响采样**: 从预训练的GMM模型采样负载相关延迟
- **非对称度控制**: 支持0-100的非对称度参数，控制上行/下行延迟比例
- **随机选择策略**: 随机选择部分数据点应用延迟，保持分布的真实性
- **RAW数据修改**: 只修改RAW时间戳列(第11-14列)，保持其他数据不变

**参数配置**:
```bash
python log_processor.py input.log output.log \
  --cpu-load 50 \          # CPU负载百分比 (0-100)
  --network-load 300 \     # 网络负载 (100-600 Mbps)
  --temp-100 45 \          # 设备100温度 (°C)
  --temp-101 65 \          # 设备101温度 (°C)
  --asymmetry 30           # 非对称度 (0-100, 50为均衡)
```

**延迟合成逻辑**:
- 选中的数据点: `t2/t3 += d_ms`, `t4 += d_ms + d_sm`
- 未选中的数据点: 保持原始值不变
- 非对称度影响: `asymmetry<50`偏向上行，`>50`偏向下行

#### 2. 对比分析器 (`simple_comparison.py`)
对比原始日志与处理后日志的延迟分布差异：

**分析指标**:
- **KS统计量**: Kolmogorov-Smirnov检验，度量分布相似性
- **KL散度**: Kullback-Leibler散度，度量分布信息差异  
- **分位数对比**: 各种百分位数的差异分析
- **CDF分析**: 累积分布函数的最大差值和位置

**可视化输出**:
- 密度直方图对比 (常规尺度 + 对数尺度)
- CDF对比图 (常规尺度 + 对数尺度)
- 统计量对比表格
- 分析报告 (.md格式)

**使用示例**:
```bash
python simple_comparison.py \
  --original_log original.log \
  --processed_log processed.log \
  --output_dir comparison_results
```

**特色功能**:
- 原始数据自动分位过滤 (1%-99%) 去除异常值
- 处理后数据保持完整，不做过滤
- 支持KL散度和KS统计量双重评估
- 自动生成详细的markdown分析报告

---

## 使用流程示例

### 1. 延迟分析流程
```bash
# 1. 分析单个日志的延迟分布
cd scripts/analysis/delayAnalyzer
python data_processor.py --input ../../../data/load_tests_0612/cpu_load/log0-original.log \
                         --output_dir ../../../results/delay_analysis \
                         --load_level 1

# 2. 批量分析多个负载条件
python load_analysis.py --input_dir ../../../data/load_tests_0612/cpu_load \
                        --output_dir ../../../results/load_comparison
```

### 2. 日志模拟流程  
```bash
# 1. 生成受负载影响的合成日志
cd scripts/simulator
python log_processor.py ../../../data/load_tests_0612/cpu_load/log0-original.log \
                        synthetic_high_load.log \
                        --cpu-load 75 --network-load 500 --asymmetry 30

# 2. 对比分析原始vs合成日志
python simple_comparison.py \
  --original_log ../../../data/load_tests_0612/cpu_load/log0-original.log \
  --processed_log synthetic_high_load.log \
  --output_dir load_effect_analysis
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
