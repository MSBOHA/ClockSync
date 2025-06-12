import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def sliding_window_kmeans(data,column, window_size=100, n_clusters=4, random_state=9, remove_outliers=False,plot = False):
    """
    对数据进行滑动窗口滤波和 KMeans 聚类划分。

    参数:
        data (pd.DataFrame): 输入数据，包含时间序列和目标列。
        column (int): 要处理的列索引。
        window_size (int): 滑动窗口大小，默认为 100。
        n_clusters (int): KMeans 聚类的簇数，默认为 4。
        random_state (int): KMeans 的随机种子，默认为 9。
        remove_outliers (bool): 是否剔除异常值，默认为 False。

    返回:
        labels (np.ndarray): 聚类标签。
        plot_data (pd.Series): 滑动窗口的均值。
        rolling_variance (pd.Series): 滑动窗口的方差。
        cluster_ranges (list of tuples): 每个聚类的横坐标范围 (start, end)。
    """
    # 滑动窗口滤波
    filtered_data = data[column].rolling(window=window_size, min_periods=1)
    plot_data = filtered_data.mean()
    rolling_variance = filtered_data.var()
    # 剔除异常值（如果启用）
    if remove_outliers:
        mean = rolling_variance.mean()
        std = rolling_variance.std()
        upper_bound = mean + 2 * std
        lower_bound = mean - 2 * std

        # 剔除异常值，将其替换为均值
        rolling_variance = rolling_variance.apply(
            lambda x: mean if x > upper_bound or x < lower_bound else x
        )
    indices = data[0]-data[0][0]
    # 准备数据用于 KMeans 聚类
    rolling_variance = rolling_variance.fillna(0)  # 填充 NaN 值
    data_for_clustering = np.column_stack((indices, rolling_variance))

    # 使用 KMeans 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state,algorithm='elkan').fit(data_for_clustering)
    labels = kmeans.labels_

    # 计算每个聚类的横坐标范围
    cluster_ranges = []
    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        if len(cluster_indices) > 0:
            cluster_ranges.append((cluster_indices[0], cluster_indices[-1]))
    sorted(cluster_ranges, key=lambda x: x[0])
    # 可视化结果
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    
    # 绘制原始数据和滑动窗口均值
    ax[0].plot(indices, data[column], label="Original Data", alpha=0.5)
    ax[0].plot(indices, plot_data, label="Rolling Mean", color="red")
    ax[0].set_title("Original Data and Rolling Mean")
    ax[0].set_xlabel("Index")
    ax[0].set_ylabel("Value")
    ax[0].legend()

    # 绘制滑动窗口方差
    ax[1].plot(indices, rolling_variance, label="Rolling Variance", color="green")
    ax[1].set_title("Rolling Variance")
    ax[1].set_xlabel("Index")
    ax[1].set_ylabel("Variance")
    ax[1].legend()

    # 绘制聚类结果
    scatter = ax[2].scatter(indices, rolling_variance, c=labels, cmap="viridis", s=10)
    ax[2].set_title("KMeans Clustering on Rolling Variance")
    ax[2].set_xlabel("Index")
    ax[2].set_ylabel("Variance")
    # plt.colorbar(scatter, ax=ax[2], label="Cluster")
    if plot:
        plt.tight_layout()
        plt.show()
    
    return labels, plot_data, rolling_variance, cluster_ranges