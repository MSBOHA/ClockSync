import numpy as np
import random
from typing import List, Tuple
import matplotlib.pyplot as plt
import time

def get_slope(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    线性回归估计斜率和标准误差
    :param x: 自变量向量
    :param y: 因变量向量
    :return: 斜率, 斜率的标准误
    """
    n = len(x)
    if n < 3:
        return 0.0, 0.0

    x = np.array(x)
    y = np.array(y)
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    cov = np.sum((x - x_mean) * (y - y_mean))
    var_x = np.sum((x - x_mean) ** 2)

    if var_x == 0:
        return 0.0, 0.0

    slope = cov / var_x
    residuals = y - (y_mean + slope * (x - x_mean))
    sse = np.sum(residuals ** 2)
    slope_se = np.sqrt(sse / (n - 2) / var_x)

    return slope, slope_se


def get_median(x: List[float]) -> float:
    """
    返回中位数
    """
    return float(np.median(x))


def get_intercept(x: List[float], y: List[float], slope: float) -> float:
    """
    中位数方法估计截距
    :param x: 自变量
    :param y: 因变量
    :param slope: 已知斜率
    :return: 截距
    """
    intercepts = [y[i] - slope * x[i] for i in range(len(x))]
    return get_median(intercepts)


class DataPoint:
    """
    时间戳结构：t1,t2,t3,t4
    """
    def __init__(self, t1: float, t2: float, t3: float, t4: float):
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4

    def sub_t2t1(self):
        return self.t2 - self.t1

    def sub_t3t4(self):
        return self.t3 - self.t4

    def avg_t1t4(self):
        return 0.5 * (self.t1 + self.t4)

    def offset(self):
        return 0.5 * (self.sub_t2t1() + self.sub_t3t4())


class SyncModel:
    """
    同步模型：估计skew和offset
    """
    def __init__(self):
        self.skew = 0.0
        self.offset = 0.0
        self.history_skews = []

    def update_sync_model(self, data: List[DataPoint], skew_smoothing_size: int = 9):
        MODEL_C2S = 0
        MODEL_OFFSET = 1
        MODEL_S2C = 2

        x = [[], [], []]
        y = [[], [], []]

        for dp in data:
            x[MODEL_C2S].append(dp.t1)
            y[MODEL_C2S].append(dp.sub_t2t1())

            x[MODEL_OFFSET].append(dp.avg_t1t4())
            y[MODEL_OFFSET].append(dp.offset())

            x[MODEL_S2C].append(dp.t4)
            y[MODEL_S2C].append(dp.sub_t3t4())

        slopes = []
        slope_ses = []

        for i in range(3):
            slope, slope_se = get_slope(x[i], y[i])
            slopes.append(slope)
            slope_ses.append(slope_se)

        best_model = int(np.argmin(slope_ses))
        best_skew = slopes[best_model]

        # 平滑处理
        if len(self.history_skews) >= skew_smoothing_size:
            self.history_skews.pop(0)
        self.history_skews.append(best_skew)
        self.skew = get_median(self.history_skews)
        # 使用偏移模型估计offset
        self.offset = get_intercept(x[MODEL_OFFSET], y[MODEL_OFFSET], self.skew)


def generate_data(real_skew: float, points: int = 100, interval: float = 1.0) -> List[DataPoint]:
    """
    生成模拟时间同步数据
    :param real_skew: 真实skew
    :param points: 数据点数
    :param interval: 时间间隔
    :return: DataPoint 列表
    """
    data = []
    freq_ratio = 1.0 + real_skew
    t = 0.0
    for _ in range(points):
        d1 = abs(random.gauss(300e-6, 100e-6))
        d2 = abs(random.gauss(300e-6, 100e-6))
        d3 = abs(random.gauss(300e-6, 100e-6))

        t1 = t
        t2 = (t + d1) * freq_ratio
        t3 = (t + d1 + d3) * freq_ratio
        t4 = t + d1 + d3 + d2

        data.append(DataPoint(t1, t2, t3, t4))
        t += interval
    return data

class TestHarness:
    """
    用于批量测试skew/offset估计的验证器
    """
    TIME_THRESH = 1e-4     # 时间误差阈值（100微秒）
    SKEW_THRESH = 1e-6     # 偏差误差阈值（1 ppm）
    DATA_POINTS = 100      # 每个测试用例数据点数
    INTERVAL = 1.0         # 间隔时间（秒）

    def __init__(self):
        self.results = []

    def execute_tests(self, total: int = 100):
        np.random.seed(int(time.time()))
        real_skews = np.clip(np.random.normal(0.0, 80e-6, total), -99e-6, 99e-6)

        for i, real_skew in enumerate(real_skews, 1):
            data = generate_data(real_skew, self.DATA_POINTS, self.INTERVAL)
            model = SyncModel()
            model.update_sync_model(data)

            est_skew = model.skew
            est_offset = model.offset
            total_time = self.DATA_POINTS * self.INTERVAL

            # 估计误差
            skew_err = real_skew - est_skew
            time_err = (real_skew * total_time) - (est_offset + est_skew * total_time)
            passed = abs(time_err) <= self.TIME_THRESH and abs(skew_err) <= self.SKEW_THRESH

            self.results.append({
                "id": i,
                "real_skew": real_skew,
                "est_skew": est_skew,
                "skew_err": skew_err,
                "time_err": time_err,
                "passed": passed
            })

    def print_statistics(self):
        print("\nValidation Report")
        print("| ID  | RealSkew   | EstSkew    | SkewErr    | TimeError  | Result |")
        print("|-----|------------|------------|------------|------------|--------|")
        for r in self.results:
            print(f"| {r['id']:3d} | {r['real_skew']:10.2e} | {r['est_skew']:10.2e} | "
                  f"{r['skew_err']:10.2e} | {r['time_err']:10.2e} | "
                  f"{'PASS' if r['passed'] else 'FAIL':6} |")

        passed = sum(1 for r in self.results if r['passed'])
        rms_skew = np.sqrt(np.mean([r['skew_err'] ** 2 for r in self.results]))
        rms_time = np.sqrt(np.mean([r['time_err'] ** 2 for r in self.results]))
        max_skew = max(abs(r['skew_err']) for r in self.results)
        max_time = max(abs(r['time_err']) for r in self.results)

        print(f"\nSummary:\nPassed: {passed}/{len(self.results)}")
        print(f"RMS Skew Error: {rms_skew * 1e6:.2f} ppm")
        print(f"Max Skew Error: {max_skew * 1e6:.2f} ppm")
        print(f"RMS Time Error: {rms_time * 1e6:.2f} µs")
        print(f"Max Time Error: {max_time * 1e6:.2f} µs")

    def plot_errors(self):
        ids = [r["id"] for r in self.results]
        skew_errs = [r["skew_err"] * 1e6 for r in self.results]  # ppm
        time_errs = [r["time_err"] * 1e6 for r in self.results]  # us

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(ids, skew_errs, marker='o', linestyle='-', label="Skew Error (ppm)")
        plt.axhline(self.SKEW_THRESH * 1e6, color='r', linestyle='--', label="Threshold")
        plt.axhline(-self.SKEW_THRESH * 1e6, color='r', linestyle='--')
        plt.xlabel("Test Case ID")
        plt.ylabel("Skew Error (ppm)")
        plt.title("Skew Estimation Error")
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(ids, time_errs, marker='x', linestyle='-', color='g', label="Time Error (us)")
        plt.axhline(self.TIME_THRESH * 1e6, color='r', linestyle='--', label="Threshold")
        plt.axhline(-self.TIME_THRESH * 1e6, color='r', linestyle='--')
        plt.xlabel("Test Case ID")
        plt.ylabel("Time Error (µs)")
        plt.title("Accumulated Time Error")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# 主程序
if __name__ == "__main__":
    tester = TestHarness()
    tester.execute_tests(total=100)
    tester.print_statistics()
    tester.plot_errors()