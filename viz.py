import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 创建图片目录
os.makedirs("pictures", exist_ok=True)

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_visualizations():
    try:
        df = pd.read_csv("data/all_videos_full_clean.csv")
    except FileNotFoundError:
        print("错误: 找不到数据文件，请先运行 clean.py")
        return

    print(f"数据形状: {df.shape}")

    # ========== 图1：播放量分布 ==========
    plt.figure(figsize=(10, 5))
    plt.hist(df["play"], bins=30, alpha=0.7, color='skyblue')
    plt.title("视频播放量分布")
    plt.xlabel("播放量")
    plt.ylabel("视频数量")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pictures/play_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 图2：点赞量 vs 播放量 ==========
    plt.figure(figsize=(10, 5))
    plt.scatter(df["play"], df["like"], s=10, alpha=0.6)
    plt.title("播放量与点赞量关系")
    plt.xlabel("播放量")
    plt.ylabel("点赞数")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pictures/play_like_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 图3：弹幕量 vs 播放量 ==========
    plt.figure(figsize=(10, 5))
    plt.scatter(df["play"], df["danmaku"], s=10, alpha=0.6, color='orange')
    plt.title("播放量与弹幕数量关系")
    plt.xlabel("播放量")
    plt.ylabel("弹幕数")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pictures/play_danmaku_scatter.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 图4：视频类型分布 ==========
    plt.figure(figsize=(12, 6))
    type_counts = df["typename"].value_counts()
    type_counts.plot(kind="bar", color='lightgreen')
    plt.title("视频类型分布")
    plt.xlabel("视频类型")
    plt.ylabel("数量")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pictures/type_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 图5：不同类型平均播放量 ==========
    plt.figure(figsize=(12, 6))
    type_play_mean = df.groupby("typename")["play"].mean().sort_values(ascending=False)
    type_play_mean.plot(kind="bar", color='coral')
    plt.title("不同视频类型的平均播放量")
    plt.xlabel("视频类型")
    plt.ylabel("平均播放量")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pictures/type_play_mean.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 图6：交互指标相关性热力图 ==========
    interact_cols = ["play", "like", "coin", "favorite", "share", "danmaku", "reply"]
    available_cols = [col for col in interact_cols if col in df.columns]

    if len(available_cols) > 1:
        corr = df[available_cols].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".2f",
                    square=True, linewidths=0.5)
        plt.title("视频交互指标相关性热力图")
        plt.tight_layout()
        plt.savefig("pictures/interaction_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("警告: 可用于相关性分析的列不足")

    print("🎉 所有可视化图表生成完毕! 保存在 pictures/ 目录")


if __name__ == "__main__":
    create_visualizations()