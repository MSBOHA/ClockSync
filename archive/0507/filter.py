import os
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns

class LogFilterPlotter:
    def __init__(self, log_file_path, lower_percent=0.05, upper_percent=0.95):
        self.log_file_path = log_file_path
        self.lower_percent = lower_percent
        self.upper_percent = upper_percent
        self.columns = [
            "ns2s(getT1(REAL))", "ns2s(getT2(REAL))", "ns2s(getT3(REAL))", "ns2s(getT4(REAL))",
            "ns2us(getT2T1(REAL))", "ns2us(getT3T4(REAL))", "ns2us(getOffset(REAL))",
            "ns2us(getCalculatedOffset(REAL))", "ns2us(getDelay(REAL))", "ns2s(getT4T1(REAL))",
            "ns2s(getT1(RAW))", "ns2s(getT2(RAW))", "ns2s(getT3(RAW))", "ns2s(getT4(RAW))",
            "ns2us(getT2T1(RAW))", "ns2us(getT3T4(RAW))", "ns2us(getOffset(RAW))",
            "ns2us(getCalculatedOffset(RAW))", "ns2us(getDelay(RAW))", "ns2s(getT4T1(RAW))",
            "seqNo",
            "ns2us(t1RealKernel-t1RealApp)",
            "ns2us(t2RealApp-t2RealKernel)",
            "ns2us(t3RealKernel-t3RealApp)",
            "ns2us(t4RealApp-t4RealKernel)"
        ]
        self.df = None
        self.mask = None

    def load_data(self):
        self.df = pd.read_csv(self.log_file_path, sep=" ", header=None)
        self.df.columns = self.columns

    def filter_data(self):
        delay_raw = self.df["ns2us(getDelay(RAW))"]
        lower = delay_raw.quantile(self.lower_percent)
        upper = delay_raw.quantile(self.upper_percent)
        self.mask = (delay_raw >= lower) & (delay_raw <= upper)

    def _plot_core(self, T1_Real, cols, mask=None, save_path="plot.svg", title=""):
        # sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        colors = ['tab:blue', 'tab:orange', 'tab:green']
        markers = ['o', 'o', 'o']
        labels = [self.columns[14], self.columns[15], self.columns[16]]

        # 绘制保留数据
        for i, col in enumerate(cols):
            if mask is not None:
                plt.plot(T1_Real[mask], col[mask], markers[i], label=labels[i] + " (kept)", color=colors[i], markersize=1, alpha=0.8)
                plt.plot(T1_Real[~mask], col[~mask], 'x', label=labels[i] + " (filtered)", color=colors[i], markersize=1, alpha=0.4)
            else:
                plt.plot(T1_Real, col, markers[i], label=labels[i], color=colors[i], markersize=1, alpha=0.8)

        plt.xlabel("T1_Real (ns2s(getT1(REAL)))")
        plt.ylabel("Value")
        plt.title(title)
        plt.legend(loc='best', fontsize=9, frameon=True)
        plt.tight_layout()
        plt.grid(True, linestyle='--', alpha=0.5)
        output_dir = os.path.dirname(save_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(save_path, dpi=300)
        plt.close()

    def plot_raw(self, save_path="plot_raw.svg"):
        T1_Real = self.df["ns2s(getT1(REAL))"]
        col15 = self.df["ns2us(getT2T1(RAW))"]
        col16 = self.df["ns2us(getT3T4(RAW))"]
        col17 = self.df["ns2us(getOffset(RAW))"]
        self._plot_core(
            T1_Real, [col15, col16, col17], mask=None, save_path=save_path,
            title="Columns 15, 16, 17 vs T1_Real (Raw, no filtering)"
        )

    def plot(self, save_path="plot_filtered.svg"):
        T1_Real = self.df["ns2s(getT1(REAL))"]
        col15 = self.df["ns2us(getT2T1(RAW))"]
        col16 = self.df["ns2us(getT3T4(RAW))"]
        col17 = self.df["ns2us(getOffset(RAW))"]
        self._plot_core(
            T1_Real, [col15, col16, col17], mask=self.mask, save_path=save_path,
            title=f"Columns 15, 16, 17 vs T1_Real\n(Filtered by Delay_Raw, percent={self.lower_percent*100:.1f}%~{self.upper_percent*100:.1f}%)"
        )

if __name__ == "__main__":
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "自有方法", "log0-original.log")
    plotter = LogFilterPlotter(log_file_path, lower_percent=0.05, upper_percent=0.95)
    plotter.load_data()
    plotter.plot_raw(save_path=os.path.join("0507", "plot_raw.svg"))
    plotter.filter_data()
    plotter.plot(save_path=os.path.join("0507", "plot_filtered.svg"))