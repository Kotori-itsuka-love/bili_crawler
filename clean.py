import pandas as pd
import numpy as np


def clean_data():
    # 读取原始爬虫数据
    df = pd.read_csv("data/all_videos_metadata.csv")

    print("原始数据形状:", df.shape)
    print("原始字段：", df.columns.tolist())

    # 转换数值字段
    num_cols = ["play", "like", "coin", "favorite", "share", "danmaku", "reply"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            print(f"警告: 列 {col} 不存在")

    # 缺失值填充（用中位数）
    for col in num_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"列 {col}: 缺失值填充为 {median_val}")

    # 去重：按 bvid + cid
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["bvid", "cid"])
    after_dedup = len(df)
    print(f"去重: {before_dedup} -> {after_dedup}")

    # 删除播放量 0 的视频
    df = df[df["play"] > 0]
    print(f"删除播放量为0后: {len(df)}")

    # 删除标题为空的视频
    df = df[df["title"].notna()]
    print(f"删除标题为空后: {len(df)}")

    # 输出清洗后的 csv
    output_file = "data/all_videos_full_clean.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("清洗完成！")
    print(f"输出文件: {output_file}")
    print(f"最终数据形状: {df.shape}")

    # 显示数据类型分布
    if 'typename' in df.columns:
        print("\n视频类型分布:")
        print(df['typename'].value_counts())

    return df


if __name__ == "__main__":
    clean_data()