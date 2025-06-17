import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 或 SimHei
plt.rcParams['axes.unicode_minus'] = False
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

def general_stats_analysis(log_file, output_dir='results', plot_real_time=False, filter_percent=0.05, piecewise_fit=False, piece_num=3, plot_raw_delay=False):
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
    if len(t1_raw) > 1:
        if piecewise_fit and piece_num > 1:
            from scipy.optimize import curve_fit
            def piecewise_linear(x, *params):
                segs = piece_num
                ks = params[:segs]
                bs = params[segs:2*segs]
                xs = [min(x)] + sorted(params[2*segs:]) + [max(x)]
                y = np.zeros_like(x)
                for i in range(segs):
                    mask = (x >= xs[i]) & (x < xs[i+1]) if i < segs-1 else (x >= xs[i]) & (x <= xs[i+1])
                    y[mask] = ks[i]*x[mask] + bs[i]
                return y
            xvals = t1_raw.values
            yvals = offset_raw.values
            k_init = [0.0]*piece_num
            b_init = [np.mean(yvals)]*piece_num
            x_init = list(np.percentile(xvals, np.linspace(0,100,piece_num+1)[1:-1]))
            p0 = k_init + b_init + x_init
            try:
                popt, _ = curve_fit(piecewise_linear, xvals, yvals, p0=p0, maxfev=10000)
                offset_est = piecewise_linear(t1_raw.values, *popt)
                plt.plot(t1_raw * 1e6, offset_est, label=f'{piece_num}段折线拟合Offset', color='black', linewidth=2, alpha=0.7)
                plt.legend()
                plt.xlabel('T1 (RAW) [us]')
                plt.ylabel('时延 [us]')
                plt.title('Raw: T2-T1、T3-T4及拟合Offset关于T1的时域图')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'raw_t2t1_t3t4_offset_vs_t1.png'))
                plt.show()
                plt.close()
                # 斜率图单独figure
                plt.figure(figsize=(8,5))
                ks = popt[:piece_num]
                x_idx = np.arange(1, piece_num+1)
                bars = plt.bar(x_idx, ks, color='skyblue', alpha=0.8, label='分段斜率')
                plt.plot(x_idx, ks, color='orange', marker='o', linewidth=2, label='斜率趋势')
                for i, k in enumerate(ks):
                    plt.text(x_idx[i], k, f'{k:.2e}', ha='center', va='bottom', fontsize=10)
                plt.xlabel('分段编号')
                plt.ylabel('Offset斜率 [us/s]')
                plt.title(f'Offset多段拟合各段斜率（{piece_num}段）')
                plt.grid(axis='y', alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'offset_piecewise_slopes.png'))
                plt.show()
                plt.close()
            except Exception as e:
                print('多段折线拟合失败，降级为全局线性拟合:', e)
        if offset_est is None:
            # 全局线性拟合
            A = np.vstack([t1_raw, np.ones_like(t1_raw)]).T
            y = offset_raw.values
            coef, resid, _, _ = np.linalg.lstsq(A, y, rcond=None)
            offset_est = coef[0] * t1_raw + coef[1]
            plt.plot(t1_raw * 1e6, offset_est, label='拟合Offset', color='black', linewidth=2, alpha=0.7)
            plt.legend()
            plt.xlabel('T1 (RAW) [us]')
            plt.ylabel('时延 [us]')
            plt.title('Raw: T2-T1、T3-T4及拟合Offset关于T1的时域图')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'raw_t2t1_t3t4_offset_vs_t1.png'))
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
    # 保存统计量为csv
    pd.DataFrame(stats_dict).T.to_csv(os.path.join(output_dir, 'delay_stats_summary.csv'))
    # 绘制统计量对比条形图，并在柱状顶部标注具体数值
    stats_df = pd.DataFrame(stats_dict).T
    plt.figure(figsize=(8,6))
    ax = plt.gca()
    stats_df[['min','q25','mean','q75','max']].plot(kind='bar', ax=ax, rot=15)
    plt.ylabel('时延 [us]')
    plt.title('自估上下行及均值时延统计量对比（min, 25%, mean, 75%, max）')
    plt.grid(axis='y', alpha=0.3)
    # 优化数字标注位置，避免重叠
    offset_list = [-0.25, -0.12, 0, 0.12, 0.25]  # 5组错开
    for i, col in enumerate(['min','q25','mean','q75','max']):
        for j, val in enumerate(stats_df[col]):
            ax.text(j + offset_list[i], val, f'{val:.2f} us', ha='center', va='bottom', fontsize=9, rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'delay_stats_comparison.png'))
    plt.show()
    plt.close()

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
    args = parser.parse_args()
    general_stats_analysis(args.log_file, output_dir=args.output_dir, plot_real_time=args.plot_real_time, filter_percent=args.filter_percent, piecewise_fit=args.piecewise_fit, piece_num=args.piece_num, plot_raw_delay=args.plot_raw_delay)
