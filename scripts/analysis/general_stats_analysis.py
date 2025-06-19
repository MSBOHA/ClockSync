import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 SimHei
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300  # 提高图片分辨率
plt.rcParams['savefig.dpi'] = 300  # 保存图片的分辨率

# 使用seaborn的美化样式
sns.set(style='whitegrid', font='Microsoft YaHei', font_scale=1.15)
sns.set_palette('Set2')  # 使用Set2色板，颜色协调且区分度高
# 严格按照README的28个字段名
def get_column_names():
    return [
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

def general_stats_analysis(log_file, output_dir='results', plot_real_time=False, filter_percent=0.05, piecewise_fit=False, piece_num=3, plot_raw_delay=False, show_plots=False):
    os.makedirs(output_dir, exist_ok=True)
    column_names = get_column_names()
    df = pd.read_csv(log_file, sep='\s+', header=None)
    if len(df.columns) >= len(column_names):
        df.columns = column_names + [f"col_{i}" for i in range(len(df.columns)-len(column_names))]
    else:
        df.columns = column_names[:len(df.columns)]


    # 根据百分位数过滤数据，支持自定义下/上分位
    lower_percent = getattr(args, 'lower_percent', 0.05) if 'args' in locals() else 0.01
    upper_percent = getattr(args, 'upper_percent', 0.95) if 'args' in locals() else 0.99
    delay_raw = df["ns2us(getDelay(RAW))"]
    q_low = delay_raw.quantile(lower_percent)
    q_high = delay_raw.quantile(upper_percent)
    print(f"分位过滤区间: {q_low:.2f} ~ {q_high:.2f} us")
    mask = (delay_raw >= q_low) & (delay_raw <= q_high)
    print("mask value_counts:\n", mask.value_counts())
    df = df[mask].reset_index(drop=True)
    print("df shape after quantile filter:", df.shape)

    # 1. RawTime: T2-T1 和 T3-T4 关于 T1 的时域图，并加入拟合offset曲线
    t1_raw = df["ns2s(getT1(RAW))"]  # 保持为秒
    t2t1_raw = df["ns2us(getT2T1(RAW))"]
    t3t4_raw = df["ns2us(getT3T4(RAW))"]
    offset_raw = df["ns2us(getOffset(RAW))"]
    plt.figure(figsize=(12,6))
    plt.plot(t1_raw * 1e6, t2t1_raw, label='T2-T1 (RAW)', alpha=0.8)
    plt.plot(t1_raw * 1e6, t3t4_raw, label='T3-T4 (RAW)', alpha=0.8)
    offset_est = None
    # 先对delay_raw做分位过滤，得到用于拟合offset的子集
    fit_lower_percent = getattr(args, 'fit_lower_percent', 0.1) if 'args' in locals() else 0.01
    fit_upper_percent = getattr(args, 'fit_upper_percent', 0.9) if 'args' in locals() else 0.3
    fit_q_low = delay_raw.quantile(fit_lower_percent)
    fit_q_high = delay_raw.quantile(fit_upper_percent)
    fit_mask = (delay_raw >= fit_q_low) & (delay_raw <= fit_q_high)
    t1_raw_fit = t1_raw[fit_mask]
    t2t1_raw_fit = t2t1_raw[fit_mask]
    t3t4_raw_fit = t3t4_raw[fit_mask]
    offset_raw_fit = offset_raw[fit_mask]
    # offset_raw 拟合目标恢复为原始offset_raw
    # 拟合时用offset_raw_fit
    # offset拟合方法选择：'svr'（默认）或'lsq'（最小二乘）
    fit_method = getattr(args, 'fit_method', 'svr') if 'args' in locals() else 'svr'  # 可选'svr'或'lsq'
    if len(t1_raw_fit) > 1:
        offset_est = np.full_like(t1_raw.values, np.nan, dtype=float)
        if piecewise_fit and piece_num > 1:
            segs = piece_num
            # 用全量原始数据分段，保证每个点都能被拟合
            xvals = t1_raw.values
            yvals = offset_raw.values
            xs = np.percentile(xvals, np.linspace(0, 100, segs + 1))
            xs[-1] = xvals.max()  # 确保最后一个边界是最大值
            for i in range(segs):
                if i == segs - 1:
                    seg_mask = (xvals >= xs[i]) & (xvals <= xs[i+1])
                else:
                    seg_mask = (xvals >= xs[i]) & (xvals < xs[i+1])
                X_seg = xvals[seg_mask]
                y_seg = yvals[seg_mask]
                print(f"分段 {i+1}/{segs}，数据点数: {len(X_seg)}")
                if len(X_seg) < 2:
                    continue
                if fit_method == 'svr':
                    from sklearn.svm import SVR
                    # 归一化
                    X_mean, X_std = X_seg.mean(), X_seg.std()
                    y_mean, y_std = y_seg.mean(), y_seg.std()
                    X_norm = (X_seg - X_mean) / X_std
                    y_norm = (y_seg - y_mean) / y_std
                    svr = SVR(kernel='linear', C=100, epsilon=0.01)
                    svr.fit(X_norm.reshape(-1, 1), y_norm)
                    # 预测并还原
                    X_all = xvals[seg_mask]
                    X_all_norm = (X_all - X_mean) / X_std
                    pred_norm = svr.predict(X_all_norm.reshape(-1, 1))
                    pred = pred_norm * y_std + y_mean
                    offset_est[seg_mask] = pred
                    print(f"分段 {i+1}/{segs} 拟合前10个: 预测={pred[:10]}, 真实={y_seg[:10]}")
                else:  # lsq
                    coef = np.polyfit(X_seg, y_seg, 1)
                    pred = coef[0] * X_seg + coef[1]
                    offset_est[seg_mask] = pred
                    print(f"分段 {i+1}/{segs} 拟合前10个: 预测={pred[:10]}, 真实={y_seg[:10]}")
        else:
            # 全局拟合直接用全部fit区间数据
            X_fit = t1_raw_fit.values
            y_fit = offset_raw_fit.values
            print(f"全局拟合，数据点数: {len(X_fit)}")
            if fit_method == 'svr':
                from sklearn.svm import SVR
                # 归一化
                X_mean, X_std = X_fit.mean(), X_fit.std()
                y_mean, y_std = y_fit.mean(), y_fit.std()
                X_norm = (X_fit - X_mean) / X_std
                y_norm = (y_fit - y_mean) / y_std
                svr = SVR(kernel='linear', C=100, epsilon=0.01)
                svr.fit(X_norm.reshape(-1, 1), y_norm)
                X_all_norm = (t1_raw.values - X_mean) / X_std
                pred_norm = svr.predict(X_all_norm.reshape(-1, 1))
                pred = pred_norm * y_std + y_mean
                offset_est = pred
                print(f"全局拟合前10个: 预测={pred[:10]}, 真实={offset_raw.values[:10]}")
            else:
                coef = np.polyfit(X_fit, y_fit, 1)
                pred = np.polyval(coef, t1_raw.values)
                offset_est = pred
                print(f"全局拟合前10个: 预测={pred[:10]}, 真实={offset_raw.values[:10]}")
    # offset拟合完成后立即画图    if 'offset_est' in locals() and offset_est is not None:
        # 计算拟合质量指标
        from sklearn.metrics import mean_squared_error, r2_score
        
        # 用原始offset_raw计算拟合质量
        mse = mean_squared_error(offset_raw, offset_est)
        rmse = np.sqrt(mse)
        r2 = r2_score(offset_raw, offset_est)
        
        plt.figure(figsize=(12,6))
        plt.plot(t1_raw * 1e6, t2t1_raw, label='T2-T1 (RAW)', alpha=0.7, marker='.', markersize=3, linestyle='-')
        plt.plot(t1_raw * 1e6, t3t4_raw, label='T3-T4 (RAW)', alpha=0.7, marker='.', markersize=3, linestyle='-')
        plt.plot(t1_raw * 1e6, offset_est, label=f'{piece_num if piecewise_fit and piece_num > 1 else "全局"}拟合Offset', 
                color='black', linewidth=2.5, alpha=0.8)
        
        # 添加拟合质量信息到图例
        fit_info = f'拟合方法: {fit_method}\nRMSE: {rmse:.4f} us\nR²: {r2:.4f}'
        plt.annotate(fit_info, xy=(0.02, 0.97), xycoords='axes fraction',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8),
                    va='top', ha='left', fontsize=10)
        
        plt.legend(loc='lower right')
        plt.xlabel('T1 (RAW) [us]')
        plt.ylabel('时延 [us]')
        plt.title('Raw: T2-T1、T3-T4及拟合Offset关于T1的时域图')
        plt.grid(True, alpha=0.3)
        
        # 保存拟合质量信息
        with open(os.path.join(output_dir, 'fit_quality.txt'), 'w', encoding='utf-8') as f:
            f.write(f"拟合方法: {fit_method}\n")
            f.write(f"拟合段数: {piece_num if piecewise_fit and piece_num > 1 else 1} (全局拟合)\n")
            f.write(f"均方根误差 (RMSE): {rmse:.6f} us\n")
            f.write(f"决定系数 (R²): {r2:.6f}\n")
            f.write(f"数据过滤区间: {fit_q_low:.2f} ~ {fit_q_high:.2f} us\n")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'raw_t2t1_t3t4_offset_vs_t1.png'), dpi=300)
        if show_plots:
            plt.show()
        plt.close()
    # 新时延（直接用原始数据）
    up_delay_est = np.array(t2t1_raw - offset_est)
    down_delay_est = np.array(offset_est - t3t4_raw)

    # 2. 可选：RealTime: T2-T1 和 T3-T4 关于 RealT1 的时域图（修正横轴）
    if plot_real_time and "ns2us(getT2T1(REAL))" in df.columns and "ns2us(getT3T4(REAL))" in df.columns:
        t1_real = df["ns2s(getT1(REAL))"]  # 保持为秒
        t2t1_real = df["ns2us(getT2T1(REAL))"]
        t3t4_real = df["ns2us(getT3T4(REAL))"]
        plt.figure(figsize=(12,6))
        plt.plot(t1_real * 1e6, t2t1_real, label='T2-T1 (REAL)', alpha=0.8)
        plt.plot(t1_real * 1e6, t3t4_real, label='T3-T4 (REAL)', alpha=0.8)
        plt.xlabel('T1 (REAL) [us]')
        plt.ylabel('时延 [us]')
        plt.title('Real: T2-T1 和 T3-T4 关于 RealT1 的时域图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'real_t2t1_t3t4_vs_real_t1.png'))
        plt.show()
        plt.close()

    # 3. 字段27和28：上下行时延的时域图和密度分布
    up_delay = df.iloc[:, 26]  # 字段27
    down_delay = df.iloc[:, 27]  # 字段28
    x = np.arange(len(up_delay))
    stats_dict = {}
    if plot_raw_delay:
        plt.figure(figsize=(12,6))
        plt.scatter(x, up_delay, label='上行时延 (字段27)', alpha=0.7, s=10)
        plt.scatter(x, down_delay, label='下行时延 (字段28)', alpha=0.7, s=10)
        plt.xlabel('样本序号')
        plt.ylabel('时延 [us]')
        plt.title('上下行时延的时域图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'up_down_delay_time.png'))
        plt.show()
        plt.close()
        # 密度分布
        for arr, name in zip([up_delay, down_delay], ['上行时延', '下行时延']):
            plt.figure(figsize=(10,5))
            arr = arr.dropna()
            mu, sigma = arr.mean(), arr.std()
            arr.plot(kind='kde', label=f'{name} 密度')
            plt.axvline(mu, color='r', linestyle='--', label=f'均值: {mu:.2f}')
            plt.axvline(mu+sigma, color='g', linestyle=':', label=f'+1σ: {mu+sigma:.2f}')
            plt.axvline(mu-sigma, color='g', linestyle=':', label=f'-1σ: {mu-sigma:.2f}')
            plt.title(f'{name} 密度分布 (均值={mu:.2f}, 标准差={sigma:.2f})')
            plt.xlabel('时延 [us]')
            plt.ylabel('密度')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{name}_delay_density.png'))
            plt.show()
            plt.close()
        # 统计量也包含原始上下行时延
        for arr, name in zip([up_delay, down_delay], ['上行时延', '下行时延']):
            arr = pd.Series(arr).dropna()
            stats = {
                'min': arr.min(),
                'max': arr.max(),
                'mean': arr.mean(),
                'std': arr.std(),
                'q25': arr.quantile(0.25),
                'q75': arr.quantile(0.25)
            }
            stats_dict[name] = stats

    # 只统计自估上下行时延和它们的均值
    stats_dict = {}
    arr_up = pd.Series(up_delay_est).dropna()
    arr_down = pd.Series(down_delay_est).dropna()
    stats_dict['自估上行时延'] = {
        'min': arr_up.min(),
        'max': arr_up.max(),
        'mean': arr_up.mean(),
        'std': arr_up.std(),
        'q25': arr_up.quantile(0.25),
        'q75': arr_up.quantile(0.75)
    }
    stats_dict['自估下行时延'] = {
        'min': arr_down.min(),
        'max': arr_down.max(),
        'mean': arr_down.mean(),
        'std': arr_down.std(),
        'q25': arr_down.quantile(0.25),
        'q75': arr_down.quantile(0.75)
    }
    # 上下行均值
    stats_dict['上下行均值'] = {k: (stats_dict['自估上行时延'][k] + stats_dict['自估下行时延'][k]) / 2 for k in stats_dict['自估上行时延']}
    # 统计量增加标准差
    # stats_dict['标准差'] = {'自估上行时延': arr_up.std(), '自估下行时延': arr_down.std(), '上下行均值': ((arr_up.std() + arr_down.std()) / 2)}
    # 保存统计量为csv
    pd.DataFrame(stats_dict).T.to_csv(os.path.join(output_dir, 'delay_stats_summary.csv'))    # 绘制统计量对比条形图，并在柱状顶部标注具体数值
    stats_df = pd.DataFrame(stats_dict).T
    plt.figure(figsize=(10,7))
    ax = plt.gca()
    
    # 自定义颜色和透明度
    colors = sns.color_palette("Set2", n_colors=5)
    stats_df[['min','q25','mean','q75','max']].plot(
        kind='bar', ax=ax, rot=0, color=colors, alpha=0.85, 
        width=0.75, edgecolor='black', linewidth=0.8
    )
    
    plt.ylabel('时延 [us]', fontsize=12)
    plt.title('自估上下行及均值时延统计量对比', fontsize=14, pad=15)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 添加图例说明
    legend_labels = ['最小值', '25%分位数', '均值', '75%分位数', '最大值']
    plt.legend(legend_labels, loc='upper right', frameon=True, 
              framealpha=0.85, edgecolor='gray')
    
    # 优化数字标注位置，避免重叠
    offset_list = [-0.25, -0.12, 0, 0.12, 0.25]  # 5组错开
    for i, col in enumerate(['min','q25','mean','q75','max']):
        for j, val in enumerate(stats_df[col]):
            if pd.notnull(val) and np.isfinite(val):
                # 为数值添加底色背景框，使数字更易读
                ax.text(j + offset_list[i], val, f'{val:.2f}', 
                        ha='center', va='bottom', fontsize=9, rotation=0,
                        bbox=dict(facecolor='white', alpha=0.7, 
                                 edgecolor='gray', boxstyle='round,pad=0.2'))
    
    # 美化x轴标签
    plt.xticks(fontsize=11, fontweight='bold')
    
    # 添加网格带背景，增强对比度
    ax.set_axisbelow(True)
    ax.set_facecolor('whitesmoke')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delay_stats_comparison.png'), dpi=300)
    
    if 'show_plots' in locals() and show_plots:
        plt.show()
    plt.close()# 绘制自估上下行时延的分布图 - 改用直方图而不是KDE
    for arr, name, color_idx in zip([arr_up, arr_down], ['自估上行时延', '自估下行时延'], [0, 1]):
        # 创建两个子图：左侧直方图，右侧箱型图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5), gridspec_kw={'width_ratios': [3, 1]})
        
        # 1. 直方图 - 显示实际数据分布
        # 使用Freedman-Diaconis规则确定bin数量，或使用固定数量
        bins = min(100, int(np.sqrt(len(arr))))  # 限制最大bin数为100
        color = sns.color_palette('Set2')[color_idx]
        counts, bins, patches = ax1.hist(arr, bins=bins, alpha=0.8, 
                                        color=color, edgecolor='black', 
                                        label=f'{name}频率分布')
        
        # 计算基本统计量
        mu, sigma = arr.mean(), arr.std()
        median = arr.median()
        q25, q75 = arr.quantile(0.25), arr.quantile(0.75)
        min_val, max_val = arr.min(), arr.max()
        
        # 计算范围，用于y轴设置
        value_range = max_val - min_val
        y_min = min_val - value_range * 0.05
        y_max = max_val + value_range * 0.05
        
        # 在直方图上标记统计量
        ax1.axvline(mu, color='r', linestyle='--', linewidth=1.5, label=f'均值: {mu:.2f}')
        ax1.axvline(median, color='blue', linestyle='-', linewidth=1.5, label=f'中位数: {median:.2f}')
        ax1.axvline(q25, color='green', linestyle=':', linewidth=1.5, label=f'25%分位: {q25:.2f}')
        ax1.axvline(q75, color='green', linestyle=':', linewidth=1.5, label=f'75%分位: {q75:.2f}')
        
        # 标注统计量
        text_y_pos = ax1.get_ylim()[1] * 0.9  # 文字位置在y轴顶部90%处
        stats_text = f'均值: {mu:.2f}\n标准差: {sigma:.2f}\n中位数: {median:.2f}\nIQR: {q75-q25:.2f}'
        ax1.text(max_val, text_y_pos, stats_text, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                ha='right', va='top', fontsize=9)
        
        # 设置直方图标题和标签
        ax1.set_title(f'{name}分布')
        ax1.set_xlabel('时延 [us]')
        ax1.set_ylabel('频次')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. 箱型图 - 直观展示分布特征
        boxplot = ax2.boxplot(arr, vert=True, patch_artist=True, 
                   boxprops=dict(facecolor=color, color='black', alpha=0.7),
                   whiskerprops=dict(color='black'),
                   capprops=dict(color='black'),
                   flierprops=dict(marker='o', markerfacecolor='red', markersize=5, 
                                  markeredgecolor='black', alpha=0.7),
                   medianprops=dict(color='red', linewidth=1.5))
        
        # 箱型图添加直观标注
        # 设置箱型图的y轴范围与直方图相似，便于比较
        ax2.set_ylim(y_min, y_max)
        
        # 在箱型图右侧添加标注文本
        for i, (val, label_text) in enumerate([
            (min_val, f"最小值: {min_val:.2f}"),
            (q25, f"Q1: {q25:.2f}"),
            (median, f"中位数: {median:.2f}"),
            (q75, f"Q3: {q75:.2f}"),
            (max_val, f"最大值: {max_val:.2f}")
        ]):
            # 在箱型图右侧添加小标签线
            ax2.annotate(
                label_text, xy=(1.05, val), xycoords=('axes fraction', 'data'),
                ha='left', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8)
            )
            
        ax2.set_title('箱型图统计')
        ax2.set_xlabel(name)
        ax2.grid(True, alpha=0.3)
        ax2.get_xaxis().set_visible(False)  # 隐藏x轴刻度
        
        # 整体布局调整
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_distribution.png'), dpi=300)
        
        # 再生成一个只有直方图的版本，方便快速查看
        plt.figure(figsize=(10,5))
        plt.hist(arr, bins=min(50, int(np.sqrt(len(arr)))), alpha=0.8, 
                color=color, edgecolor='black')
        plt.axvline(mu, color='r', linestyle='--', linewidth=1.5, label=f'均值: {mu:.2f}')
        plt.axvline(median, color='blue', linestyle='-', linewidth=1.5, label=f'中位数: {median:.2f}')
        plt.title(f'{name}直方图')
        plt.xlabel('时延 [us]')
        plt.ylabel('频次')
        
        # 添加更详细的统计信息框
        stats_text = (f'均值: {mu:.2f} us\n'
                     f'中位数: {median:.2f} us\n'
                     f'标准差: {sigma:.2f} us\n'
                     f'25%分位数: {q25:.2f} us\n'
                     f'75%分位数: {q75:.2f} us\n'
                     f'IQR: {q75-q25:.2f} us')
        plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes,
                bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'),
                ha='right', va='top', fontsize=10)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_histogram.png'), dpi=300)
          # 添加ECDF图（经验累积分布函数）- 对时延分析很有用
        plt.figure(figsize=(10,5))
        # 计算ECDF
        x = np.sort(arr.values)
        y = np.arange(1, len(arr) + 1) / len(arr)
        
        plt.plot(x, y, marker='.', markersize=5, linestyle='-', color=color, linewidth=2, alpha=0.8)
        plt.axhline(y=0.5, color='r', linestyle='--', label='中位数水平 (50%)')
        plt.axhline(y=0.9, color='g', linestyle='--', label='90%水平')
        plt.axhline(y=0.99, color='purple', linestyle='--', label='99%水平')
        
        # 标记关键百分位点
        percentiles = [50, 90, 95, 99]
        percentile_values = np.percentile(arr, percentiles)
        
        for p, val in zip(percentiles, percentile_values):
            plt.plot([val, val], [0, p/100], 'k--', alpha=0.5)
            plt.plot([0, val], [p/100, p/100], 'k--', alpha=0.5)
            plt.annotate(f'{p}%: {val:.2f}', 
                        xy=(val, p/100), xytext=(10, 0), 
                        textcoords='offset points', ha='left', va='center',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.8))
        
        plt.title(f'{name} - 累积分布函数 (ECDF)')
        plt.xlabel('时延 [us]')
        plt.ylabel('累积概率')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_ecdf.png'), dpi=300)
        
        if 'show_plots' in locals() and show_plots:
            plt.show()
        plt.close('all')




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='General Log Stats Analysis')
    parser.add_argument('log_file', help='输入log文件路径')
    parser.add_argument('--output_dir', default='results/general_stats', help='输出图片目录')
    parser.add_argument('--plot_real_time', action='store_true', help='是否绘制RealTime的T2-T1和T3-T4')
    parser.add_argument('--filter_percent', type=float, default=0.05, help='过滤极值百分比(0~0.5)')
    parser.add_argument('--piecewise_fit', action='store_true', help='是否使用多段折线拟合Offset')
    parser.add_argument('--piece_num', type=int, default=3, help='多段折线拟合的段数')
    parser.add_argument('--lower_percent', type=float, default=0.01, help='下分位过滤百分比(0~1)')
    parser.add_argument('--upper_percent', type=float, default=0.99, help='上分位过滤百分比(0~1)')
    parser.add_argument('--plot_raw_delay', action='store_true', help='是否输出原始字段27/28上下行时延相关图')
    parser.add_argument('--fit_lower_percent', type=float, default=0.1, help='拟合Offset时下分位过滤百分比(0~1)')
    # parser.add_argument('--fit_upper_percent', type=float, default=0.9, help='拟合Offset时上分位过滤百分比(0~1)')
    parser.add_argument('--fit_method', default='svr', help="offset拟合方法：'svr'（默认）或'lsq'（最小二乘）")
    args = parser.parse_args()
    general_stats_analysis(args.log_file, output_dir=args.output_dir, plot_real_time=args.plot_real_time, filter_percent=args.filter_percent, piecewise_fit=args.piecewise_fit, piece_num=args.piece_num, plot_raw_delay=args.plot_raw_delay,show_plots = False)
