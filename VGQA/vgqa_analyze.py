#!/usr/bin/env python3
"""
VGQA 评测结果分析脚本
通过 prompt-q/o CSV 将模型的 q1...q6 答案映射到 index.json 的问题 index，
直接输出除 index 23“视频内容是否具有美感？”外的 31 个问题级子属性分数，
不再合并为动作、场景、整体等上层维度。
"""

import json
import csv
import numpy as np
import pandas as pd
from collections import defaultdict
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_FILE = os.environ.get(
    "VGQA_INPUT_FILE", os.path.join(PROJECT_ROOT, "vgqa_pred_results.jsonl")
)
OUTPUT_DIR = os.environ.get("VGQA_SCORE_OUTPUT_DIR", SCRIPT_DIR)
INDEX_JSON = os.path.join(SCRIPT_DIR, "index.json")
PROMPT_QO_CSV = os.path.join(SCRIPT_DIR, "prompt_qo_deduped.csv")

QOS = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6']
EXCLUDED_INDEXES = {23}  # “视频内容是否具有美感？”不计入 31 个子属性


def build_prompt_to_index_mapping():
    """
    读取 index.json 和推理使用的 prompt-q/o CSV，建立
    prompt -> {q_key: index} 映射。
    """
    with open(INDEX_JSON, 'r', encoding='utf-8') as f:
        index_list = json.load(f)

    # question text -> index
    q2idx = {}
    # index -> {question, options}
    idx_info = {}
    for item in index_list:
        q2idx[item['question']] = item['index']
        idx_info[item['index']] = {
            'question': item['question'],
            'options': item['options'],
        }

    # 推理和评分共用同一份 q/o 配置，保证 q1...q6 的含义一致。
    prompt_qmap = {}
    df = pd.read_csv(PROMPT_QO_CSV)
    for _, row in df.iterrows():
        prompt = row['prompt']
        q_mapping = {}
        for qo in QOS:
            qt = row.get(qo, '')
            if pd.isna(qt) or str(qt).strip() == '':
                continue
            qt = str(qt).strip()
            if qt in q2idx:
                q_mapping[qo] = q2idx[qt]
        prompt_qmap[prompt] = q_mapping

    return prompt_qmap, idx_info


def parse_score(value_str):
    """从 gpt 答案中提取数值分数，格式如 '3：描述' 或 '-1：描述'"""
    if '：' in value_str:
        score_str = value_str.split('：')[0]
    elif ':' in value_str:
        score_str = value_str.split(':')[0]
    else:
        score_str = value_str[0]
    return float(score_str)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 建立映射
    prompt_qmap, idx_info = build_prompt_to_index_mapping()

    # 2. 读取 JSONL，按 index 收集分数
    index_scores = defaultdict(list)
    total_videos = 0
    error_count = 0
    na_count = 0
    unmatched_prompts = set()

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            d = json.loads(line.strip())
            if 'error' in d:
                error_count += 1
                continue
            total_videos += 1
            if 'gpt' not in d:
                continue

            prompt = d['prompt']
            if prompt not in prompt_qmap:
                unmatched_prompts.add(prompt)
                continue

            q_mapping = prompt_qmap[prompt]
            for q_key, value in d['gpt'].items():
                score = parse_score(value)
                if score == -1:
                    na_count += 1
                if q_key in q_mapping:
                    idx = q_mapping[q_key]
                    if idx not in EXCLUDED_INDEXES:
                        index_scores[idx].append(score)

    if unmatched_prompts:
        print(f"警告: {len(unmatched_prompts)} 个 prompt 未在 CSV 中找到映射")

    # 3. 按 index 计算原始均分和归一化均分
    index_results = {}
    for idx in sorted(index_scores.keys()):
        scores = np.array(index_scores[idx])
        raw_avg = float(scores.mean())

        info = idx_info.get(idx, {})
        options = info.get('options', [])

        # 用 options 首尾值作为归一化范围（与 analysics.py 一致）
        if options:
            range_min = float(options[0].split('：')[0]) if '：' in options[0] else float(options[0][0])
            range_max = float(options[-1].split('：')[0]) if '：' in options[-1] else float(options[-1][0])
        else:
            range_min = int(scores.min())
            range_max = int(scores.max())

        if range_max > range_min:
            norm_avg = (raw_avg - range_min) / (range_max - range_min)
        else:
            norm_avg = 1.0

        index_results[idx] = {
            'question': info.get('question', ''),
            'count': len(scores),
            'raw_avg': round(raw_avg, 4),
            'range_min': range_min,
            'range_max': range_max,
            'norm_avg': round(norm_avg, 4),
        }

    # 4. 直接输出 index.json 中的 31 个子属性，不再合并为上层维度。
    attribute_results = {
        idx: result for idx, result in index_results.items()
        if idx not in EXCLUDED_INDEXES
    }

    # 5. 打印结果
    print(f"总视频数: {total_videos}  |  错误数: {error_count}  |  N/A 回答数: {na_count}")
    print()
    print(f"{'index':<6} {'数量':>6} {'原始均分':>8} {'范围':>6}  {'归一化均分':>10}  子属性问题")
    print("-" * 80)
    for idx in sorted(attribute_results.keys()):
        r = attribute_results[idx]
        print(f"{idx:<6} {r['count']:>6} {r['raw_avg']:>8.4f} {r['range_min']}-{r['range_max']:>2}   {r['norm_avg']:>10.4f}  {r['question'][:40]}")

    # 31 个子属性等权计算 VGQA 总分
    all_norm = [r['norm_avg'] for r in attribute_results.values()]
    if all_norm:
        overall_norm = round(float(np.mean(all_norm)), 4)
        print(f"\nVGQA 总体归一化均分（31 个子属性等权）: {overall_norm}")

    # 6. 保存 CSV
    csv_path = os.path.join(OUTPUT_DIR, 'vgqa_dim_scores.csv')
    with open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', '属性问题', '数量', '原始均分', '归一化均分'])
        for idx in sorted(attribute_results.keys()):
            r = attribute_results[idx]
            writer.writerow([idx, r['question'], r['count'], r['raw_avg'], r['norm_avg']])
    print(f"\nCSV 已保存到: {csv_path}")

    # 7. 保存详细 JSON
    json_path = os.path.join(OUTPUT_DIR, 'vgqa_dim_scores.json')
    output = {
        'summary': {
            'total_videos': total_videos,
            'error_count': error_count,
            'na_count': na_count,
            'overall_norm_avg': overall_norm if all_norm else None,
        },
        'attributes': {str(k): v for k, v in attribute_results.items()},
        'excluded_indexes': sorted(EXCLUDED_INDEXES),
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"JSON 已保存到: {json_path}")


if __name__ == '__main__':
    main()
