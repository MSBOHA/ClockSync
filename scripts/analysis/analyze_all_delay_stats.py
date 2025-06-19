import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# 直接用项目根目录下的路径
csv_path = os.path.join('results', 'batch_stats', 'all_delay_stats_summary.csv')
df = pd.read_csv(csv_path, dtype=str)  # 全部按字符串读入，后续再做类型转换

# 解析等级
# cpu等级: log0-original_cpu1 ~ log0-original_cpu4
# 流量等级: log0-original_100M ~ log0-original_600M

def parse_level(log_name):
    if 'cpu' in log_name:
        return 'cpu', int(log_name.split('cpu')[-1])
    elif 'M' in log_name:
        return 'net', int(log_name.split('_')[-1].replace('M','').replace('.log',''))
    else:
        return 'other', -1

# 增加等级信息
levels = df['log_name'].apply(parse_level)
df['type'] = [t for t, v in levels]
df['level'] = [v for t, v in levels]

# 只分析上下行均值、上行std、下行std
mean_df = df[df['Unnamed: 1'] == '上下行均值'][['log_name','type','level','mean']].copy()
up_std_df = df[df['Unnamed: 1'] == '自估上行时延'][['log_name','type','level','std']].copy()
up_std_df.rename(columns={'std': 'std_upstd'}, inplace=True)
down_std_df = df[df['Unnamed: 1'] == '自估下行时延'][['log_name','type','level','std']].copy()

# 类型转换
mean_df['mean'] = pd.to_numeric(mean_df['mean'], errors='coerce')
up_std_df['std_upstd'] = pd.to_numeric(up_std_df['std_upstd'], errors='coerce')
down_std_df['std'] = pd.to_numeric(down_std_df['std'], errors='coerce')

# 合并
stat_df = mean_df.merge(up_std_df, on=['log_name','type','level'])
stat_df = stat_df.merge(down_std_df, on=['log_name','type','level'])
stat_df.rename(columns={'std':'down_std'}, inplace=True)

# 中文兼容及绘图全局设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
plt.rcParams['figure.dpi'] = 300  # 提高图片分辨率
plt.rcParams['savefig.dpi'] = 300  # 保存图片的分辨率
plt.rcParams['font.size'] = 12  # 默认字体大小
plt.rcParams['axes.labelsize'] = 13  # 轴标签字体大小
plt.rcParams['axes.titlesize'] = 14  # 标题字体大小
plt.rcParams['xtick.labelsize'] = 11  # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 11  # y轴刻度标签字体大小

# 新建保存目录
save_dir = os.path.join('results', 'batch_stats', 'delay_stats_figs')
os.makedirs(save_dir, exist_ok=True)

# 设置seaborn样式
sns.set(style='whitegrid', font='SimHei', font_scale=1.15)
sns.set_palette('Set2')  # 使用Set2色板，颜色协调且区分度高

# 美化后的绘图函数

def plot_bar_with_error(stat_df, stat, stat_label, factor_type, save_name):
    sub = stat_df[stat_df['type'] == factor_type].copy()
    # 获取等级作为字符串标签
    x_vals = sub['level'].astype(int).values
    x_pos = np.arange(len(x_vals))  # 位置索引
    y = sub[stat]
    
    # 确定误差值 - 使用标准差
    if stat == 'mean':
        # 限制误差棒大小，确保不会显示为负值
        yerr = sub['std_upstd'] if 'std_upstd' in sub.columns else None
        if yerr is not None:
            # 确保下误差棒不会超过均值本身(不显示负值)
            yerr_lower = np.minimum(y.values, yerr.values)
            yerr = np.vstack([yerr_lower, yerr.values])  # [下误差棒, 上误差棒]
    else:
        yerr = None  # 其他统计量暂不绘制误差棒
        
    # 创建更优雅的图表
    plt.figure(figsize=(10, 6))
    
    # 设置背景色和网格样式
    ax = plt.gca()
    ax.set_axisbelow(True)  # 网格线置于数据下方
    ax.set_facecolor('#f8f8f8')  # 设置轻微的背景色
    
    # 选择颜色
    color_idx = 0 if factor_type == 'cpu' else 1
    colors = sns.color_palette('Set2')
    main_color = colors[color_idx]
    
    # 调整柱状图宽度和外观
    bar_width = 0.65
    
    # 美化误差棒
    error_kw = {
        'capsize': 6,
        'capthick': 2,
        'ecolor': 'black',
        'elinewidth': 1.8,
        'alpha': 0.8
    }
    
    # 绘制柱状图
    bars = plt.bar(x_pos, y, color=main_color, alpha=0.85, width=bar_width, 
            yerr=yerr, error_kw=error_kw, edgecolor='black', linewidth=0.8)
    
    # 设置自定义标签
    if factor_type == 'cpu':
        labels = [f"CPU {x*25}%" for x in x_vals]  # CPU标签: CPU 25%, CPU 50%...
    else:
        labels = [f"{x}M" for x in x_vals]      # 网络标签: 100M, 200M...
        
    # 设置轴标签和标题，增加字体大小和粗细
    plt.xlabel('CPU负载等级' if factor_type=='cpu' else '流量等级(Mbps)', fontsize=13)
    plt.ylabel(stat_label, fontsize=13)
    plt.title(f"{('CPU' if factor_type=='cpu' else '网络')}负载 - {stat_label}", 
              fontsize=15, fontweight='bold', pad=15)
    plt.xticks(x_pos, labels, fontsize=12)
    
    # 在柱状图顶部添加数值标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.02 * max(y)),
                f'{height:.2f}', ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='gray', alpha=0.8))
    
    # 添加网格线但仅限y轴
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 添加边框增强对比度
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(0.8)
        
    # 紧凑布局
    plt.tight_layout()
    
    # 保存高质量图片
    plt.savefig(os.path.join(save_dir, f'{save_name}_{factor_type}.png'), dpi=300)
    plt.close()

# 2x1画布分别绘制CPU和网络负载的均值柱状图

def plot_mean_vs_level_subplot(stat_df, save_name):
    # 创建更美观的子图布局，增加间距
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'hspace': 0.3})
    
    # 设置整体风格
    for ax in axes:
        ax.set_axisbelow(True)  # 网格线置于数据下方
        ax.set_facecolor('#f8f8f8')  # 设置轻微的背景色
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.8)
    
    # CPU数据处理
    cpu = stat_df[stat_df['type']=='cpu'].copy()
    x_cpu_vals = cpu['level'].astype(int).values
    x_pos_cpu = np.arange(len(x_cpu_vals))
    y_cpu = cpu['mean']
    yerr_cpu = cpu['std_upstd']  # 使用上行标准差作为误差棒
    
    # 网络数据处理
    net = stat_df[stat_df['type']=='net'].copy()
    x_net_vals = net['level'].astype(int).values
    x_pos_net = np.arange(len(x_net_vals))
    y_net = net['mean']
    yerr_net = net['std_upstd']  # 使用上行标准差作为误差棒
    
    # 处理误差棒，防止显示为负值
    yerr_cpu_upper = yerr_cpu.values.copy()
    yerr_cpu_lower = np.minimum(y_cpu.values, yerr_cpu.values)  # 不要显示低于0的误差棒
    yerr_cpu_plot = np.vstack([yerr_cpu_lower, yerr_cpu_upper])  # [下误差棒, 上误差棒]
    
    yerr_net_upper = yerr_net.values.copy()
    yerr_net_lower = np.minimum(y_net.values, yerr_net.values)  # 不要显示低于0的误差棒
    yerr_net_plot = np.vstack([yerr_net_lower, yerr_net_upper])  # [下误差棒, 上误差棒]
    
    # 计算两个图表的最大值，用于统一y轴
    max_value = max(
        np.max(y_cpu + yerr_cpu) if len(y_cpu) > 0 else 0,
        np.max(y_net + yerr_net) if len(y_net) > 0 else 0
    )
    y_max = max_value * 1.15  # 留出15%的空间
    
    # 美化误差棒样式
    error_kw = {
        'capsize': 6,
        'capthick': 2,
        'ecolor': 'black',
        'elinewidth': 1.8,
        'alpha': 0.8
    }
    
    # 设置CPU子图
    cpu_bars = axes[0].bar(x_pos_cpu, y_cpu, color=sns.color_palette('Set2')[0], 
               alpha=0.85, width=0.65, yerr=yerr_cpu_plot, error_kw=error_kw,
               edgecolor='black', linewidth=0.8)
    
    # 自定义CPU标签
    cpu_labels = [f"{x*25}%" for x in x_cpu_vals]  # 25%, 50%, 75%, 100%
    axes[0].set_xlabel('CPU负载等级', fontsize=13)
    axes[0].set_ylabel('上下行时延均值 [us]', fontsize=13)
    axes[0].set_title('CPU负载 - 上下行时延均值', fontsize=15, fontweight='bold', pad=10)
    axes[0].set_xticks(x_pos_cpu)
    axes[0].set_xticklabels(cpu_labels, fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].set_ylim(0, y_max)  # 设置统一的y轴上限
    
    # 添加数值标签
    for i, bar in enumerate(cpu_bars):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + (0.02 * max(y_cpu)),
                f'{height:.2f}', ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='gray', alpha=0.8))
    
    # 设置网络子图
    net_bars = axes[1].bar(x_pos_net, y_net, color=sns.color_palette('Set2')[1], 
               alpha=0.85, width=0.65, yerr=yerr_net_plot, error_kw=error_kw,
               edgecolor='black', linewidth=0.8)
    
    # 自定义网络标签
    net_labels = [f"{x}M" for x in x_net_vals]  # 100M, 200M, 300M...
    axes[1].set_xlabel('流量等级(Mbps)', fontsize=13)
    axes[1].set_ylabel('上下行时延均值 [us]', fontsize=13)
    axes[1].set_title('网络负载 - 上下行时延均值', fontsize=15, fontweight='bold', pad=10)
    axes[1].set_xticks(x_pos_net)
    axes[1].set_xticklabels(net_labels, fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].set_ylim(0, y_max)  # 设置统一的y轴上限
    
    # 添加数值标签
    for i, bar in enumerate(net_bars):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + (0.02 * max(y_net)),
                f'{height:.2f}', ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor='gray', alpha=0.8))
    
    # 添加总标题
    fig.suptitle('CPU与网络负载对时延均值的影响对比', fontsize=16, fontweight='bold', y=0.98)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 保留顶部空间给suptitle
    
    # 保存高质量图片
    plt.savefig(os.path.join(save_dir, save_name), dpi=300)
    plt.close()

# 绘制CPU
plot_bar_with_error(stat_df, 'mean', '上下行时延均值', 'cpu', 'mean_bar')
plot_bar_with_error(stat_df, 'std_upstd', '上行时延标准差', 'cpu', 'upstd_bar')
plot_bar_with_error(stat_df, 'down_std', '下行时延标准差', 'cpu', 'downstd_bar')
# 绘制网络
plot_bar_with_error(stat_df, 'mean', '上下行时延均值', 'net', 'mean_bar')
plot_bar_with_error(stat_df, 'std_upstd', '上行时延标准差', 'net', 'upstd_bar')
plot_bar_with_error(stat_df, 'down_std', '下行时延标准差', 'net', 'downstd_bar')
# 2x1画布分别绘制均值
plot_mean_vs_level_subplot(stat_df, 'stats_mean_vs_level.png')

# 探究哪个影响因素大：用线性回归斜率绝对值对比
results = []
for stat in ['mean','std_upstd','down_std']:
    for t in ['cpu','net']:
        sub = stat_df[stat_df['type']==t]
        X = sub['level'].values.reshape(-1,1)
        y = sub[stat].values
        if len(X) > 1:
            model = LinearRegression().fit(X, y)
            slope = abs(model.coef_[0])
            r_squared = model.score(X, y)  # 拟合优度R²
            results.append({
                'stat': stat, 
                'type': t, 
                'slope': slope,
                'r_squared': r_squared
            })

result_df = pd.DataFrame(results)
print('各统计量对CPU/流量等级的敏感性（斜率绝对值越大影响越大）:')
print(result_df)

# 保存结果
result_df.to_csv(os.path.join('results', 'batch_stats', 'stats_sensitivity.csv'), index=False)

# 绘制敏感性分析结果的条形图
plt.figure(figsize=(12, 8))
ax = plt.gca()
ax.set_axisbelow(True)  # 网格线置于数据下方
ax.set_facecolor('#f8f8f8')  # 设置轻微的背景色

# 美化边框
for spine in ax.spines.values():
    spine.set_edgecolor('gray')
    spine.set_linewidth(0.8)

title_map = {
    'mean': '上下行时延均值', 
    'std_upstd': '上行时延标准差', 
    'down_std': '下行时延标准差'
}

# 将数据重塑为两组：CPU影响和网络影响
cpu_data = result_df[result_df['type'] == 'cpu'].set_index('stat')['slope']
net_data = result_df[result_df['type'] == 'net'].set_index('stat')['slope']

# 获取R²值用于标注
r2_cpu = result_df[result_df['type'] == 'cpu'].set_index('stat')['r_squared'].to_dict()
r2_net = result_df[result_df['type'] == 'net'].set_index('stat')['r_squared'].to_dict()

# 创建DataFrame便于绘图
plot_df = pd.DataFrame({
    'CPU负载敏感度': cpu_data,
    '网络负载敏感度': net_data
})

# 用更友好的中文标签
plot_df.index = [title_map[idx] for idx in plot_df.index]

# 绘制条形图
bar_plot = plot_df.plot(kind='bar', figsize=(12, 8), width=0.75, 
                 color=[sns.color_palette('Set2')[0], sns.color_palette('Set2')[1]],
                 edgecolor='black', linewidth=0.8, ax=ax)

# 添加数值标签并包含R²信息
for i, p in enumerate(ax.patches):
    # 计算标签位置和要显示的信息
    height = p.get_height()
    
    # 确定当前柱子对应的统计量和类型
    stat_idx = i // 2
    is_cpu = i % 2 == 0
    
    # 获取对应的统计量键
    stat_key = list(title_map.keys())[stat_idx]
    
    # 获取对应的R²值
    r2 = r2_cpu.get(stat_key, 0) if is_cpu else r2_net.get(stat_key, 0)
    
    # 构建标签文本
    label_text = f'斜率: {height:.4f}\nR^2: {r2:.3f}'
    
    # 创建带有多行文本和背景的注释
    ax.annotate(label_text,
                xy=(p.get_x() + p.get_width() / 2., height),
                xytext=(0, 3),  # 垂直偏移3点
                textcoords="offset points", 
                ha='center', va='bottom', 
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.9))

# 美化标题和标签
plt.title('时延统计量对CPU/网络负载的敏感性分析', fontsize=16, fontweight='bold', pad=15)
plt.ylabel('敏感度系数（线性回归斜率绝对值）', fontsize=14)
plt.xlabel('统计量类型', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 美化图例
plt.legend(title='影响因素', title_fontsize=12, 
           frameon=True, framealpha=0.9, edgecolor='gray',
           fontsize=12, loc='best')

# 添加解释文本
explanation = "注: 敏感度系数为线性回归斜率绝对值，值越大表示该因素对统计量影响越显著。\nR^2为拟合优度，值越接近1表示拟合效果越好。"
plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round,pad=0.5', fc='#f0f0f0', ec='gray', alpha=0.9))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为底部文本留出空间
plt.savefig(os.path.join(save_dir, 'sensitivity_analysis.png'), dpi=300)

# 生成结论
conclusions = []
for stat in title_map.keys():
    cpu_slope = cpu_data.get(stat, 0)
    net_slope = net_data.get(stat, 0)
    cpu_r2 = r2_cpu.get(stat, 0)
    net_r2 = r2_net.get(stat, 0)
    
    # 计算比率
    if cpu_slope > net_slope:
        ratio = cpu_slope / net_slope if net_slope > 0 else float('inf')
        conclusion = f"{title_map[stat]}对CPU负载的敏感性是网络负载的{ratio:.2f}倍"
        dominant = "CPU负载"
    else:
        ratio = net_slope / cpu_slope if cpu_slope > 0 else float('inf')
        conclusion = f"{title_map[stat]}对网络负载的敏感性是CPU负载的{ratio:.2f}倍"
        dominant = "网络负载"
    
    # 添加R²信息和更详细的解读
    detail = (f"- {title_map[stat]}:\n"
             f"  * CPU负载敏感度: {cpu_slope:.6f} (R² = {cpu_r2:.3f})\n"
             f"  * 网络负载敏感度: {net_slope:.6f} (R² = {net_r2:.3f})\n"
             f"  * 结论: {conclusion}\n"
             f"  * 主导因素: {dominant}\n")
    
    conclusions.append(detail)

# 将结论保存到格式化的文本文件
with open(os.path.join(save_dir, 'sensitivity_conclusions.txt'), 'w', encoding='utf-8') as f:
    f.write("=========================================\n")
    f.write("      时延统计量敏感性分析结论报告       \n")
    f.write("=========================================\n\n")
    f.write("【分析结果摘要】\n")
    f.write('\n'.join(conclusions))
    f.write("\n\n【敏感性分析说明】\n")
    f.write("1. 敏感性系数是线性回归拟合得到的斜率绝对值，值越大表示该因素对统计量的影响越显著\n")
    f.write("2. R²为拟合优度，值越接近1表示线性拟合效果越好，说明影响关系越稳定\n")
    f.write("3. 主导因素是指对该统计量影响更大的负载类型\n\n")
    f.write("【分析时间】\n")
    from datetime import datetime
    f.write(f"生成日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("=========================================\n")

print('分析完成，图表和敏感性结果已保存到:', save_dir)
print('敏感性分析结论报告已生成: sensitivity_conclusions.txt')
