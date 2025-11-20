import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import gc
import time
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 确保目录存在
os.makedirs("data", exist_ok=True)

DANMAKU_FILE = "data/all_danmaku.csv"
OUT_FILE = "data/video_sentiment_summary.csv"
PROGRESS_FILE = "data/progress_tracker.pkl"

# 保持原有模型
MODEL_NAME = "uer/roberta-base-finetuned-jd-binary-chinese"


def setup_device():
    """设置设备，优先使用GPU"""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("使用GPU进行推理")
    else:
        device = "cpu"
        logger.info("使用CPU进行推理（速度较慢）")
    return device


device = setup_device()

print("正在加载BERT模型...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
except Exception as e:
    logger.error(f"模型加载失败: {e}")
    exit(1)


# 进度跟踪
def load_progress():
    if os.path.exists(PROGRESS_FILE):
        return pd.read_pickle(PROGRESS_FILE)
    return {"processed_rows": 0}


def save_progress(progress):
    pd.to_pickle(progress, PROGRESS_FILE)


# 优化的情感预测函数
@torch.no_grad()
def predict_sentiment_batch(texts, batch_size=8):
    """批量预测情感，自动调整批次大小"""
    texts = [text for text in texts if isinstance(text, str) and text.strip()]
    if not texts:
        return np.array([])

    all_probs = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        try:
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(device)

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())

            # 清理内存
            del inputs, outputs
            if device == "cuda":
                torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e) and batch_size > 1:
                logger.warning(f"GPU内存不足，减小批次大小从{batch_size}到{batch_size // 2}")
                return predict_sentiment_batch(texts, batch_size=batch_size // 2)
            else:
                logger.error(f"推理失败: {e}")
                # 返回中性情感作为fallback
                all_probs.extend([0.5] * len(batch_texts))

    return np.array(all_probs)


def process_danmaku_sentiment():
    """处理弹幕情感分析"""
    # 检查进度
    progress = load_progress()
    logger.info(f"从进度点继续: 已处理 {progress['processed_rows']} 行")

    # 读取弹幕数据
    if not os.path.exists(DANMAKU_FILE):
        logger.error(f"弹幕文件不存在: {DANMAKU_FILE}")
        return

    logger.info("读取弹幕数据...")
    try:
        df = pd.read_csv(DANMAKU_FILE)
        df = df.dropna(subset=["text"])
        logger.info(f"弹幕数量: {len(df)}")
    except Exception as e:
        logger.error(f"读取弹幕文件失败: {e}")
        return

    # 如果已有进度，跳过已处理的行
    start_idx = progress["processed_rows"]
    if start_idx > 0:
        df = df.iloc[start_idx:]
        logger.info(f"跳过前 {start_idx} 行，剩余 {len(df)} 行待处理")

    # 分批处理
    chunk_size = 2000  # 减小批次大小避免内存问题
    all_sentiments = []

    for chunk_start in tqdm(range(0, len(df), chunk_size), desc="处理弹幕"):
        chunk_end = min(chunk_start + chunk_size, len(df))
        chunk = df.iloc[chunk_start:chunk_end]

        # 预测情感
        chunk_sentiments = predict_sentiment_batch(chunk["text"].tolist(), batch_size=8)
        all_sentiments.extend(chunk_sentiments)

        # 更新进度
        progress["processed_rows"] = start_idx + chunk_end
        save_progress(progress)

        # 清理内存
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # 添加延迟避免过热
        time.sleep(0.5)

        # 每处理几个块就输出进度
        if (chunk_start // chunk_size) % 10 == 0:
            logger.info(f"进度: {progress['processed_rows']}/{start_idx + len(df)}")

    # 确保长度匹配
    if len(all_sentiments) < len(df):
        all_sentiments.extend([0.5] * (len(df) - len(all_sentiments)))

    df["sentiment"] = all_sentiments[:len(df)]

    # 生成汇总统计
    logger.info("生成情感特征汇总...")
    summary_list = []

    for cid, group in df.groupby("cid"):
        if len(group) < 5:  # 忽略弹幕太少的视频
            continue

        arr = group.sort_values("time")
        s = arr["sentiment"].values

        summary_list.append({
            "cid": cid,
            "avg_sentiment": float(s.mean()),
            "std_sentiment": float(s.std()),
            "max_sentiment": float(s.max()),
            "min_sentiment": float(s.min()),
            "num_comments": len(s),
            "peak_count": int(sum(
                (s[1:-1] > s[:-2]) & (s[1:-1] > s[2:])
            )),
        })

    # 保存结果
    summary_df = pd.DataFrame(summary_list)

    # 合并已有结果
    if os.path.exists(OUT_FILE):
        existing_df = pd.read_csv(OUT_FILE)
        # 移除重复的cid
        existing_df = existing_df[~existing_df["cid"].isin(summary_df["cid"])]
        combined_df = pd.concat([existing_df, summary_df], ignore_index=True)
        combined_df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
        logger.info(f"合并结果，现有 {len(combined_df)} 个视频的情感分析")
    else:
        summary_df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
        logger.info(f"保存 {len(summary_df)} 个视频的情感分析")

    # 清理进度文件
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

    logger.info("情感分析完成!")
    return summary_df


# 运行主函数
if __name__ == "__main__":
    try:
        result = process_danmaku_sentiment()
        print("\n🎉 情感分析完成！结果已保存到：", OUT_FILE)
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        print("❌ 处理失败，但进度已保存，可以重新运行继续处理")