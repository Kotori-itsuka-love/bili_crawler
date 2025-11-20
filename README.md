📌 Bilibili 热门视频数据挖掘与弹幕情感分析系统

Bilibili Hot Videos Data Mining & Danmaku Sentiment Analysis

📖 项目简介

本项目实现了对 Bilibili 热门视频数据的一站式采集、处理、可视化和知识挖掘。
包含以下核心功能：

🔍 爬取热门视频元数据（播放、点赞、投币、收藏、分享等）

💬 批量爬取视频的弹幕

🤖 基于中文 BERT 的弹幕情感分析

📊 互动特征/KMeans 聚类/相关性分析可视化

📈 自动生成洞察与知识挖掘结果

项目最终用于课程《数据挖掘》大作业 / Web 全栈实践 / 可视化分析项目。

🚀 功能特点
✔ 视频数据爬取

支持一次性抓取 100+ 热门视频

多字段完整保存（播放、点赞、投币、收藏、分享、弹幕、评论）

多 P（多 CID）自动解析

✔ 弹幕爬取

单视频可达数万条弹幕

自动清洗、去重、文本过滤

✔ BERT 情感分析

使用 uer/roberta-base-finetuned-jd-binary-chinese

输出情绪值（0-1）+ 情绪波动度 + 高峰次数（peak_count）

✔ 数据清洗与融合

合并 CID 级指标到视频级（bvid）

数值字段类型校正

处理缺失值、异常值

✔ 数据挖掘与可视化

播放/点赞/弹幕散点图

热力图（互动特征相关性）

KMeans 聚类

情绪分布直方图

弹幕情绪气泡图、排行榜

🏗 技术栈
模块	使用技术
爬虫	requests、B站 API、WBI 鉴权、Cookies
数据处理	pandas、numpy
情感分析	transformers、torch、BERT
聚类分析	scikit-learn
可视化	matplotlib、seaborn
环境管理	Anaconda

🔧 环境安装
conda create -n bili_crawler python=3.10
conda activate bili_crawler

pip install -r requirements.txt


若出现 huggingface 下载慢，可使用镜像：

set HF_ENDPOINT=https://hf-mirror.com

🕹 使用方法
1️⃣ 爬取热门视频数据
python crawler/main.py


生成文件：

data/all_videos_full_clean.csv

data/all_danmaku.csv

2️⃣ 运行情感分析（BERT）
python analysis/bert_sentiment_viz.py


输出：

data/video_sentiment_summary.csv

一系列图表保存至 bert_pictures/

3️⃣ 运行可视化（播放/互动分析）
python analysis/viz.py


输出图表至：

pictures/

📘 数据挖掘成果（摘要）

播放量与点赞/弹幕呈显著正相关

情绪峰值（peak_count）和互动率有强相关性

KMeans 发现 3 类视频互动模式（爆款/小众高互动/高播放低参与）

弹幕总体情绪偏正向，争议性视频在情绪波动图中易区分
