# 重构后的时钟同步分析工具

这是一个重构后的时钟同步分析工具，采用模块化设计，具有以下特点：

## 主要功能

1. **数据处理**: 自动加载和清洗时钟同步日志数据
2. **Offset拟合**: 支持SVR和最小二乘法，可选分段拟合
3. **GMM拟合**: 对时延数据进行高斯混合模型拟合，支持对数变换
4. **统计分析**: 计算全面的统计量和分布特征
5. **可视化**: 生成时间序列图、分布图、ECDF图、GMM拟合图等
6. **批量处理**: 支持单文件和批量文件处理

## 模块结构

- `main.py`: 主程序入口和分析器类
- `data_processor.py`: 数据加载、清洗和拟合功能
- `statistics.py`: 统计量计算
- `visualizer.py`: 图表生成
- `config.py`: 配置管理
- `file_manager.py`: 文件输出管理

## 使用方法

### 单文件分析
```bash
python main.py data/log_file.log --output_dir results/analysis
```

### 批量分析目录
```bash
python main.py data/load_tests_0612/ --batch_mode --output_dir results/batch_analysis
```

### 启用GMM拟合
```bash
python main.py data/log_file.log --gmm_max_components 8 --output_dir results/gmm_analysis
```

### 分段拟合
```bash
python main.py data/log_file.log --piecewise_fit --piece_num 5 --fit_method svr
```

## 参数说明

### 数据过滤
- `--lower_percent`: 下分位过滤阈值 (默认: 0.01)
- `--upper_percent`: 上分位过滤阈值 (默认: 0.99)

### 拟合设置
- `--fit_method`: 拟合方法 (svr/lsq, 默认: svr)
- `--piecewise_fit`: 启用分段拟合
- `--piece_num`: 分段数量 (默认: 3)

### GMM设置
- `--disable_gmm`: 禁用GMM拟合
- `--gmm_min_components`: GMM最小组件数 (默认: 1)
- `--gmm_max_components`: GMM最大组件数 (默认: 6)
- `--no_gmm_log_transform`: 禁用对数变换

### 输出设置
- `--output_dir`: 输出目录
- `--no_save_details`: 不保存详细数据
- `--show_plots`: 显示图表

## 输出结构

```
results/
├── stats/              # 统计摘要
├── plots/              # 图表文件
├── gmm_params/         # GMM参数 (JSON + Pickle)
├── raw_data/           # 原始数据导出
└── reports/            # 综合报告
```

对于批量处理，每个负载等级会创建独立的子目录：
```
results/
├── load_400M/
├── load_cpu1/
└── ...
```

## 新增功能

1. **GMM拟合**: 
   - 自动选择最优组件数 (基于BIC)
   - 支持对数变换处理偏态分布
   - 保存完整的GMM参数供后续使用

2. **负载等级识别**:
   - 自动从文件名提取负载等级 (如400M, cpu1等)
   - 为不同负载等级创建独立的分析结果

3. **增强的统计分析**:
   - 偏度和峰度
   - 变异系数
   - 平均绝对偏差
   - 四分位距系数

4. **改进的可视化**:
   - GMM拟合结果图
   - 增强的分布图 (含正态拟合曲线)
   - 更好的图表标注和信息展示

## 依赖包

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy

## 示例用法

```python
from refactored import ClockSyncAnalyzer, AnalysisConfig

# 创建配置
config = AnalysisConfig(
    enable_gmm=True,
    gmm_log_transform=True,
    piecewise_fit=True,
    piece_num=5
)

# 创建分析器
analyzer = ClockSyncAnalyzer(config)

# 分析单个文件
result = analyzer.analyze_single_file('data/log_file.log')

# 批量分析
file_list = ['file1.log', 'file2.log', 'file3.log']
batch_results = analyzer.analyze_batch(file_list)
```
