import os
import time
import json
import requests
import pandas as pd
from lxml import etree
from tqdm import tqdm
from typing import List, Dict, Optional

# -----------------------
# 配置
# -----------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}

RANKING_API = "https://api.bilibili.com/x/web-interface/ranking?rid=0&day=7&type=1&arc_type=0"
VIDEO_VIEW_API = "https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
DANMAKU_XML_URL = "https://comment.bilibili.com/{cid}.xml"

OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

# 黑名单 - 需要过滤的作者和关键词
BLACKLIST_AUTHORS = ["宋浩老师", "宋浩", "浩老师"]
BLACKLIST_KEYWORDS = ["高等数学", "高数", "微积分", "线性代数", "概率论"]


# -----------------------
# 工具函数：重试请求
# -----------------------
def safe_get(url: str, params: dict = None, headers: dict = None, timeout: int = 12, max_retries: int = 3):
    headers = headers or HEADERS
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            wait = 1 + attempt * 2
            print(f"[WARN] 请求失败 {url} （尝试 {attempt + 1}/{max_retries}），原因：{e}，{wait}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"请求失败：{url}")


# -----------------------
# 检查是否在黑名单中
# -----------------------
def is_blacklisted(title: str, author: str) -> bool:
    """检查视频是否应该被过滤"""
    title_lower = title.lower() if title else ""
    author_lower = author.lower() if author else ""

    # 检查作者黑名单
    for black_author in BLACKLIST_AUTHORS:
        if black_author.lower() in author_lower:
            return True

    # 检查标题关键词黑名单
    for keyword in BLACKLIST_KEYWORDS:
        if keyword.lower() in title_lower:
            return True

    return False


# -----------------------
# 获取排行榜（返回视频条目列表）
# -----------------------
def get_ranking_list() -> List[Dict]:
    resp = safe_get(RANKING_API)
    js = resp.json()
    if js.get("code") != 0:
        raise RuntimeError("获取排行榜失败: " + str(js))
    data = js["data"]["list"]
    # 只保留我们需要的字段
    items = []
    for item in data:
        items.append({
            "bvid": item.get("bvid"),
            "aid": item.get("aid"),
            "title": item.get("title"),
            "author": item.get("owner", {}).get("name"),
            "play": item.get("play"),
            "video_review": item.get("video_review"),  # 评论数
            "favorites": item.get("favorites"),
            "duration": item.get("duration"),
            "typename": item.get("tname"),
        })
    return items


# -----------------------
# 根据 bvid 获取视频信息（主要是 cid）
# -----------------------
def get_video_info_by_bvid(bvid: str) -> Dict:
    url = VIDEO_VIEW_API.format(bvid=bvid)
    resp = safe_get(url)
    js = resp.json()
    if js.get("code") != 0:
        raise RuntimeError(f"获取视频信息失败: {bvid} -> {js}")
    data = js["data"]

    cid = data.get("cid")
    pages = data.get("pages", [])
    page_info = []
    for p in pages:
        page_info.append({
            "cid": p.get("cid"),
            "page": p.get("page"),
            "part": p.get("part")
        })

    return {
        "bvid": bvid,
        "title": data.get("title"),
        "cid": cid,
        "pages": page_info,
        "stat": data.get("stat", {}),
        "tname": data.get("tname"),
        "owner": data.get("owner", {})
    }


# -----------------------
# 抓取并解析弹幕 xml（返回 list of dict: time, text）
# -----------------------
def fetch_and_parse_danmaku(cid: int) -> List[Dict]:
    url = DANMAKU_XML_URL.format(cid=cid)
    resp = safe_get(url)
    # xml 里弹幕文本在 <d p="...">弹幕内容</d>
    # p 属性里包含 "time,mode,size,color,..."，time 为出弹时间（秒）
    content = resp.content
    root = etree.fromstring(content)
    res = []
    for d in root.findall("d"):
        p_attr = d.get("p")
        text = d.text or ""
        if not p_attr:
            continue
        try:
            parts = p_attr.split(",")
            appear_time = float(parts[0])  # 单位：秒（小数）
        except Exception:
            appear_time = None
        res.append({
            "cid": cid,
            "time": appear_time,
            "text": text.strip()
        })
    return res


# -----------------------
# 主流程：抓取前 N 个视频及弹幕
# -----------------------
def crawl_top_n_and_save(n: int = 150, per_video_limit_danmaku: Optional[int] = None, max_parts_per_video: int = 3):
    print(f"[INFO] 获取排行榜前{n}个视频...")
    ranking = get_ranking_list()

    # 如果请求的视频数量超过实际可用的数量，则使用最大可用数量
    if len(ranking) < n:
        print(f"[INFO] 排行榜只有{len(ranking)}个视频，将爬取所有可用视频")
        n = len(ranking)

    ranking = ranking[:n]
    meta_rows = []
    all_danmaku_rows = []

    # 统计信息
    stats = {
        'total_processed': 0,
        'blacklisted': 0,
        'failed': 0,
        'success': 0
    }

    for item in tqdm(ranking, desc="Videos"):
        bvid = item["bvid"]
        stats['total_processed'] += 1

        # 第一步过滤：检查排行榜中的基本信息
        if is_blacklisted(item["title"], item["author"]):
            print(f"[FILTER] 过滤黑名单视频: {item['title']} (作者: {item['author']})")
            stats['blacklisted'] += 1
            continue

        try:
            info = get_video_info_by_bvid(bvid)
        except Exception as e:
            print(f"[ERROR] 获取 {bvid} 信息失败：{e}")
            stats['failed'] += 1
            continue

        # 第二步过滤：检查详细视频信息
        video_title = info.get("title", "")
        video_author = info.get("owner", {}).get("name", "")

        if is_blacklisted(video_title, video_author):
            print(f"[FILTER] 过滤黑名单视频: {video_title} (作者: {video_author})")
            stats['blacklisted'] += 1
            continue

        pages = info.get("pages") or []

        # 限制每个视频的最大分P数
        if len(pages) > max_parts_per_video:
            print(f"[INFO] 视频 {bvid} 有 {len(pages)} 个分P，只取前{max_parts_per_video}个")
            pages = pages[:max_parts_per_video]

        if pages:
            for p in pages:
                cid = p.get("cid")
                stat = info.get("stat", {})

                meta = {
                    "bvid": bvid,
                    "title": info.get("title"),
                    "part": p.get("part"),
                    "page": p.get("page"),
                    "cid": p.get("cid"),
                    "typename": info.get("tname"),
                    "author": info.get("owner", {}).get("name", "未知UP主"),
                    "play": stat.get("view"),
                    "like": stat.get("like"),
                    "coin": stat.get("coin"),
                    "favorite": stat.get("favorite"),
                    "share": stat.get("share"),
                    "danmaku": stat.get("danmaku"),
                    "reply": stat.get("reply"),
                }

                meta_rows.append(meta)

                # 抓弹幕
                try:
                    danmaku_list = fetch_and_parse_danmaku(cid)
                    if per_video_limit_danmaku:
                        danmaku_list = danmaku_list[:per_video_limit_danmaku]
                    all_danmaku_rows.extend(danmaku_list)

                    # 写每个 video-part 的 csv（可选）
                    part_csv = os.path.join(OUT_DIR, f"{bvid}_p{p.get('page')}_danmaku.csv")
                    pd.DataFrame(danmaku_list).to_csv(part_csv, index=False, encoding="utf-8-sig")
                    time.sleep(0.8)
                except Exception as e:
                    print(f"[WARN] 获取弹幕失败 bvid={bvid} cid={cid}：{e}")
                    time.sleep(1)

        else:
            # 仅单 cid 情况
            cid = info.get("cid")
            stat = info.get("stat", {})

            meta = {
                "bvid": bvid,
                "title": info.get("title"),
                "part": None,
                "page": 1,
                "cid": cid,
                "typename": info.get("tname"),
                "author": info.get("owner", {}).get("name", "未知UP主"),
                "play": stat.get("view"),
                "like": stat.get("like"),
                "coin": stat.get("coin"),
                "favorite": stat.get("favorite"),
                "share": stat.get("share"),
                "danmaku": stat.get("danmaku"),
                "reply": stat.get("reply"),
            }

            meta_rows.append(meta)
            try:
                danmaku_list = fetch_and_parse_danmaku(cid)
                if per_video_limit_danmaku:
                    danmaku_list = danmaku_list[:per_video_limit_danmaku]
                all_danmaku_rows.extend(danmaku_list)
                part_csv = os.path.join(OUT_DIR, f"{bvid}_danmaku.csv")
                pd.DataFrame(danmaku_list).to_csv(part_csv, index=False, encoding="utf-8-sig")
                time.sleep(0.8)
            except Exception as e:
                print(f"[WARN] 获取弹幕失败 bvid={bvid} cid={cid}：{e}")
                time.sleep(1)

        stats['success'] += 1

    # 保存元数据与合并弹幕表
    meta_df = pd.DataFrame(meta_rows)
    meta_csv = os.path.join(OUT_DIR, "all_videos_metadata.csv")
    meta_df.to_csv(meta_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存视频元数据到 {meta_csv}，共{len(meta_rows)}个视频")

    danmaku_df = pd.DataFrame(all_danmaku_rows)
    danmaku_csv = os.path.join(OUT_DIR, "all_danmaku.csv")
    danmaku_df.to_csv(danmaku_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存全部弹幕到 {danmaku_csv}，共{len(all_danmaku_rows)}条弹幕")

    # 输出统计信息
    print(f"\n[STATS] 爬取统计:")
    print(f"  - 总共处理: {stats['total_processed']}")
    print(f"  - 成功爬取: {stats['success']}")
    print(f"  - 过滤黑名单: {stats['blacklisted']}")
    print(f"  - 失败: {stats['failed']}")

    print("[DONE] 爬取完成。")


# -----------------------
# 如果作为脚本运行
# -----------------------
if __name__ == "__main__":
    # 抓取150个视频，每个视频最多3个分P，过滤宋浩老师的高等数学视频
    crawl_top_n_and_save(n=150, per_video_limit_danmaku=None, max_parts_per_video=3)