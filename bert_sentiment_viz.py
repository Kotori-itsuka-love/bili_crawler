import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 创建输出目录
os.makedirs("bert_pictures", exist_ok=True)

# 设置字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def create_sentiment_visualizations():
    # 数据路径
    DATA_PATH = "data/video_sentiment_summary.csv"

    if not os.path.exists(DATA_PATH):
        print(f"错误: 找不到情感分析结果文件 {DATA_PATH}")
        print("请先运行 sentiment_and_timeseries.py")
        return

    # 读取情感分析结果
    df = pd.read_csv(DATA_PATH)
    print(f"情感分析数据形状: {df.shape}")

    # ============ 图表 1：视频平均情绪排序 ============
    plt.figure(figsize=(12, 8))
    # 取前20个视频显示
    df_sorted = df.nlargest(20, "avg_sentiment")

    plt.barh(range(len(df_sorted)), df_sorted["avg_sentiment"],
             color=plt.cm.viridis(np.linspace(0, 1, len(df_sorted))))
    plt.yticks(range(len(df_sorted)), [f"视频 {cid}" for cid in df_sorted["cid"]])
    plt.xlabel("平均情绪得分")
    plt.ylabel("视频")
    plt.title("视频情绪得分排行榜 (Top 20)")
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig("bert_pictures/avg_sentiment_rank.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 图表 2：情绪离散度 ============
    plt.figure(figsize=(10, 6))
    plt.scatter(df["num_comments"], df["std_sentiment"],
                s=50, alpha=0.6, c=df["avg_sentiment"], cmap='viridis')
    plt.colorbar(label='平均情绪得分')
    plt.xlabel("弹幕数量")
    plt.ylabel("情绪波动（标准差）")
    plt.title("弹幕数量 vs 情绪波动")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bert_pictures/sentiment_std_vs_comments.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 图表 3：情绪最高 vs 最低 ============
    plt.figure(figsize=(10, 6))
    plt.scatter(df["max_sentiment"], df["min_sentiment"],
                s=50, alpha=0.6, c=df["num_comments"], cmap='plasma')
    plt.colorbar(label='弹幕数量')
    plt.xlabel("最高情绪得分")
    plt.ylabel("最低情绪得分")
    plt.title("视频情绪最大值与最小值关系图")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bert_pictures/sentiment_max_vs_min.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 图表 4：弹幕峰值次数 vs 平均情绪 ============
    plt.figure(figsize=(10, 6))
    plt.scatter(df["peak_count"], df["avg_sentiment"],
                s=50, alpha=0.6, c=df["std_sentiment"], cmap='coolwarm')
    plt.colorbar(label='情绪波动')
    plt.xlabel("弹幕峰值次数")
    plt.ylabel("平均情绪得分")
    plt.title("弹幕峰值次数 vs 平均情绪得分")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bert_pictures/peak_vs_avg_sentiment.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 图表 5：总结图（多变量可视化） ============
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(df["num_comments"], df["avg_sentiment"],
                          s=df["peak_count"] * 2 + 10,  # 气泡大小代表峰值数量
                          c=df["std_sentiment"],  # 颜色代表情绪波动
                          cmap='RdYlBu', alpha=0.7)

    plt.colorbar(scatter, label='情绪波动')
    plt.xlabel("弹幕数量")
    plt.ylabel("平均情绪得分")
    plt.title("弹幕数量 – 情绪得分 – 峰值次数 – 情绪波动 关系图")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bert_pictures/bubble_sentiment_overview.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ============ 图表 6：情绪分布直方图 ============
    plt.figure(figsize=(10, 6))
    plt.hist(df["avg_sentiment"], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel("平均情绪得分")
    plt.ylabel("视频数量")
    plt.title("视频平均情绪得分分布")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("bert_pictures/sentiment_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("所有情感分析图表已生成，保存在：bert_pictures/")


if __name__ == "__main__":
    create_sentiment_visualizations()