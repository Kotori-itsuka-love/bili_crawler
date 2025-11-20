# kmeans_analysis_plots.py
# 说明：读取 cluster_result.csv，生成 5 张支持性图表并保存到当前目录

import pandas as pd
import numpy as np
import matplotlib

# 使用非交互后端，避免弹窗与字体阻塞
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# 创建输出目录
os.makedirs("kmeans-pictures", exist_ok=True)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def create_kmeans_analysis():
    # 读取文件
    fn = "cluster_result.csv"
    if not os.path.exists(fn):
        raise FileNotFoundError(f"{fn} 未找到，请先运行 kmeans.py")

    df = pd.read_csv(fn)
    print(f"数据形状: {df.shape}")
    print(f"数据列: {df.columns.tolist()}")

    # 检查cluster列是否存在
    if 'cluster' not in df.columns:
        raise RuntimeError("数据中缺少 'cluster' 列，请先运行 kmeans.py")

    # 需要的数值特征列
    numeric_features = ["play", "like", "coin", "favorite", "share", "danmaku", "reply"]

    # 确保所有数值列都存在且是数值类型
    available_features = []
    for feat in numeric_features:
        if feat in df.columns:
            # 确保列是数值类型
            df[feat] = pd.to_numeric(df[feat], errors='coerce')
            # 处理无限值和NaN值
            df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
            df[feat] = df[feat].fillna(df[feat].median())
            available_features.append(feat)

    if len(available_features) < 2:
        raise RuntimeError("可用于分析的数值特征列不足")

    print(f"使用的数值特征: {available_features}")

    # 计算 per-play 比例（密度）
    for feat in available_features[1:]:  # 跳过play本身
        # 避免除零
        df[f"{feat}_per_play"] = np.where(df["play"] > 0, df[feat] / df["play"], 0)

    # 计算互动密度 - 只使用存在的数值列
    interaction_features = [f for f in available_features if f != "play"]
    if interaction_features:
        df["interaction_density"] = df[interaction_features].sum(axis=1) / np.where(df["play"] > 0, df["play"], 1)
    else:
        df["interaction_density"] = 0

    # 只选择数值列进行聚类分析
    numeric_cols = available_features + [f"{feat}_per_play" for feat in available_features[1:]] + [
        "interaction_density"]

    # 关键修复：只对数值列进行分组计算均值
    cluster_means = df.groupby("cluster")[numeric_cols].mean()

    print("聚类均值计算完成")
    print(f"聚类数量: {len(cluster_means)}")

    # ------------------ 图1：聚类平均互动密度 ------------------
    plt.figure(figsize=(8, 6))
    clusters = [str(c) for c in cluster_means.index]
    vals = cluster_means["interaction_density"].values
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    bars = plt.bar(clusters, vals, color=colors[:len(clusters)])
    plt.title("各聚类的平均互动密度")
    plt.ylabel("互动密度")
    plt.xlabel("聚类编号")

    # 在柱子上添加数值
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                 f'{val:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("kmeans-pictures/cluster_interaction_density.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ------------------ 图2：箱线图 ------------------
    ratio_cols = [f"{feat}_per_play" for feat in available_features[1:4]]  # 取前几个特征
    labels_mapping = {
        "like_per_play": "点赞/播放",
        "danmaku_per_play": "弹幕/播放",
        "reply_per_play": "评论/播放",
        "coin_per_play": "投币/播放",
        "favorite_per_play": "收藏/播放",
        "share_per_play": "分享/播放"
    }

    # 只使用存在的比例列
    available_ratio_cols = [col for col in ratio_cols if col in df.columns and len(df[col].dropna()) > 0]

    if available_ratio_cols:
        n_plots = min(4, len(available_ratio_cols))
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        clusters_sorted = sorted(df["cluster"].unique())
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']

        for i, col in enumerate(available_ratio_cols[:4]):  # 最多4个子图
            if i >= len(axes):
                break

            arrs = [df[df["cluster"] == c][col].dropna().values for c in clusters_sorted]
            # 检查是否有数据
            if all(len(arr) > 0 for arr in arrs):
                # 使用 patch_artist=True 来创建可填充的箱线图
                box = axes[i].boxplot(arrs, tick_labels=[str(c) for c in clusters_sorted], patch_artist=True)
                axes[i].set_title(f"{labels_mapping.get(col, col)}")
                axes[i].set_xlabel("聚类编号")
                axes[i].set_ylabel(labels_mapping.get(col, col))

                # 修复：使用正确的方法设置箱线图颜色
                for j, patch in enumerate(box['boxes']):
                    patch.set_facecolor(colors[j % len(colors)])
                    patch.set_alpha(0.7)
            else:
                axes[i].text(0.5, 0.5, f"无足够数据\n({labels_mapping.get(col, col)})",
                             ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f"{labels_mapping.get(col, col)}")

        # 隐藏多余的子图
        for i in range(len(available_ratio_cols), 4):
            axes[i].set_visible(False)

        fig.tight_layout()
        plt.savefig("kmeans-pictures/boxplots_per_play_ratios.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("警告: 没有可用的比例列用于箱线图")

    # ------------------ 图3：分享率 ------------------
    if "share_per_play" in cluster_means.columns:
        plt.figure(figsize=(8, 6))
        share_rates = cluster_means["share_per_play"].values
        bars = plt.bar(clusters, share_rates, color='lightcoral')
        plt.title("各聚类的平均分享率")
        plt.xlabel("聚类编号")
        plt.ylabel("分享率")

        for bar, val in zip(bars, share_rates):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(share_rates) * 0.01,
                     f'{val:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig("kmeans-pictures/share_rate_per_cluster.png", dpi=300, bbox_inches='tight')
        plt.close()

    # ------------------ 图4：雷达图 ------------------
    try:
        centers_raw = cluster_means[available_features].copy()

        # 避免除零错误和NaN值
        data_range = centers_raw.max() - centers_raw.min()
        data_range = data_range.replace(0, 1)  # 如果全相同，设为1避免除零
        centers_norm = (centers_raw - centers_raw.min()) / data_range

        categories = {
            "play": "播放量", "like": "点赞", "coin": "投币",
            "favorite": "收藏", "share": "分享", "danmaku": "弹幕", "reply": "评论"
        }
        category_names = [categories.get(f, f) for f in available_features]

        N = len(category_names)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)

        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        for idx, row in centers_norm.iterrows():
            values = row.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=f"聚类 {idx}", color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])

        ax.set_thetagrids(np.degrees(angles[:-1]), category_names)
        ax.set_title("聚类中心雷达图", size=14, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig("kmeans-pictures/cluster_centers_radar.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"雷达图生成失败: {e}")

    # ------------------ 图5：Log-Log 图 ------------------
    plt.figure(figsize=(10, 8))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    clusters_unique = sorted(df["cluster"].unique())

    has_data = False
    for i, c in enumerate(clusters_unique):
        sub = df[df["cluster"] == c]
        x = sub["play"].values
        y = sub["reply"].values
        mask = (x > 0) & (y > 0)
        x = x[mask]
        y = y[mask]

        if len(x) == 0:
            continue

        plt.scatter(x, y, label=f"聚类 {c}", alpha=0.7, color=colors[i % len(colors)])
        has_data = True

        # 幂律拟合
        try:
            if len(x) > 2:  # 需要至少3个点进行拟合
                a, b = np.polyfit(np.log(x), np.log(y), 1)
                xs = np.linspace(x.min(), x.max(), 100)
                ys = np.exp(a) * xs ** b
                plt.plot(xs, ys, linestyle='--', color=colors[i % len(colors)])
                plt.text(xs[-1], ys[-1], f"b={b:.2f}", fontsize=9, color=colors[i % len(colors)])
        except Exception as e:
            print(f"聚类 {c} 的幂律拟合失败: {e}")

    if has_data:
        plt.xscale('log')
        plt.yscale('log')
        plt.title("播放量与评论数的幂律关系")
        plt.xlabel("播放量（log）")
        plt.ylabel("评论数（log）")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("kmeans-pictures/loglog_play_reply.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("警告: 没有足够数据生成Log-Log图")
        plt.close()

    # 保存聚类均值
    cluster_means.to_csv("cluster_means_with_ratios.csv", index=True)

    print("所有分析图表已生成:")
    print(" - cluster_interaction_density.png")
    print(" - boxplots_per_play_ratios.png")
    print(" - share_rate_per_cluster.png")
    print(" - cluster_centers_radar.png")
    print(" - loglog_play_reply.png")
    print("聚类统计已保存至: cluster_means_with_ratios.csv")


if __name__ == "__main__":
    create_kmeans_analysis()