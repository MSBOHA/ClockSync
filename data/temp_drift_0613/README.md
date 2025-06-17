# 数据目录说明

本目录包含 2024年6月13日采集的 101 号设备的时钟频率与温度漂移实验原始数据。

## 文件说明
- log0-original101.log：101号设备的原始时钟数据日志。
- cpu_temp101.log：101号设备的CPU温度记录。

## 数据格式
- log0-original101.log：无表头，空格分隔，字段含义详见主目录 README 或分析脚本注释。
- cpu_temp101.log：三列，分别为采集索引、温度值（°C）、单位。

## 用途
建议配合 scripts/analysis/clock_allan_variance.py clock_freq_vs_temp.py  进行时钟频率漂移、温度变化与 Allan 方差分析。
