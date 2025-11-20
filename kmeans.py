import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# 创建输出目录
os.makedirs("pictures", exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def perform_clustering():
    try:
        df = pd.read_csv("data/all_videos_full_clean.csv")
    except FileNotFoundError:
        print("错误: 找不到数据文件，请先运行 clean.py")
        return

    print(f"数据形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")

    # 选择用于聚类的特征 - 确保都是数值列
    features = ["play", "like", "coin", "favorite", "share", "danmaku", "reply"]
    available_features = [f for f in features if f in df.columns]

    # 确保所有特征都是数值类型
    for feat in available_features:
        df[feat] = pd.to_numeric(df[feat], errors='coerce')

    if len(available_features) < 2:
        print("错误: 可用于聚类的特征不足")
        return

    print(f"使用的特征: {available_features}")
    X = df[available_features]

    # 检查并处理无限值和NaN值
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 使用肘部法则确定最佳聚类数
    print("正在计算最佳聚类数...")
    inertias = []
    K_range = range(1, min(8, len(X) + 1))  # 确保k不超过样本数

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    # 绘制肘部法则图
    plt.figure(figsize=(8, 5))
    plt.plot(K_range, inertias, 'bo-')
    plt.xlabel('聚类数量')
    plt.ylabel('簇内平方和')
    plt.title('肘部法则 - 确定最佳聚类数')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pictures/elbow_method.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 自动选择聚类数 - 找肘点
    if len(inertias) > 1:
        # 计算二阶差分找拐点
        diff1 = np.diff(inertias)
        diff2 = np.diff(diff1)
        if len(diff2) > 0:
            elbow_point = np.argmin(diff2) + 2  # +2 因为两次差分
            n_clusters = min(max(2, elbow_point), 5)  # 限制在2-5之间
        else:
            n_clusters = 3
    else:
        n_clusters = min(3, len(X))

    print(f"使用 {n_clusters} 个聚类")

    # 执行KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # 输出聚类中心（反标准化）
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers, columns=available_features)
    print("\n=== 聚类中心（反标准化） ===")
    print(centers_df)

    # 保存结果 - 包含所有原始列和聚类结果
    output_df = df.copy()
    output_df.to_csv("cluster_result.csv", index=False, encoding="utf-8-sig")

    print(f"\n聚类已完成，结果已保存至 cluster_result.csv")
    print(f"聚类分布: {df['cluster'].value_counts().sort_index()}")

    # 可视化（使用播放量 & 点赞数 作为二维显示）
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df["play"], df["like"], c=df["cluster"],
                          cmap="viridis", alpha=0.6, s=50)
    plt.xlabel("播放量")
    plt.ylabel("点赞量")
    plt.title("K-Means 视频聚类（按播放量 & 点赞）")
    plt.colorbar(scatter, label='聚类')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pictures/kmeans_result.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 额外可视化：不同类型在各聚类中的分布
    if 'typename' in df.columns:
        plt.figure(figsize=(12, 6))
        cross_tab = pd.crosstab(df['typename'], df['cluster'])
        cross_tab.plot(kind='bar', figsize=(12, 6))
        plt.title('不同类型在聚类中的分布')
        plt.xlabel('视频类型')
        plt.ylabel('数量')
        plt.xticks(rotation=45)
        plt.legend(title='聚类')
        plt.tight_layout()
        plt.savefig("pictures/type_cluster_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    print("\n聚类图已保存为 pictures/kmeans_result.png")

    return df


if __name__ == "__main__":
    perform_clustering()