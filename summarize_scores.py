#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汇总三个评测任务的维度分数，生成综合CSV
"""

import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.environ.get('MODEL_NAME', 'unnamed_model')

# 文件路径
VAQA_CSV = os.environ.get(
    'VAQA_SCORE_CSV', os.path.join(SCRIPT_DIR, 'VAQA', 'vaqa_dim_scores.csv')
)
VGQA_CSV = os.environ.get(
    'VGQA_SCORE_CSV', os.path.join(SCRIPT_DIR, 'VGQA', 'vgqa_dim_scores.csv')
)
VTAG_CSV = os.environ.get(
    'VTAG_SCORE_CSV', os.path.join(SCRIPT_DIR, 'VTAG', 'vtag_dim_scores.csv')
)
OUTPUT_DIR = os.environ.get('SUMMARY_OUTPUT_DIR', SCRIPT_DIR)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'dimension_scores_summary.csv')


def read_vaqa_scores():
    """读取VAQA维度分数"""
    scores = {}
    with open(VAQA_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dim_en = row['维度(英文)']
            avg_score = float(row['平均分'])
            scores[dim_en] = avg_score
    return scores


def read_vgqa_scores():
    """读取 VGQA 的 index 问题级子属性分数。"""
    scores = {}
    with open(VGQA_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            index = row['index']
            question = row['属性问题']
            norm_avg = float(row['归一化均分'])
            scores[f"index {index}: {question}"] = norm_avg
    return scores


def read_vtag_scores():
    """读取VTAG属性分数"""
    scores = {}
    with open(VTAG_CSV, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            attr = row['属性']
            if attr == '总体':
                continue
            acc = float(row['准确率']) / 100.0  # 转换为0-1范围
            scores[attr] = acc
    return scores


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("开始汇总各维度分数")
    print("=" * 60)

    # 读取各维度分数
    vaqa_scores = read_vaqa_scores()
    vgqa_scores = read_vgqa_scores()
    vtag_scores = read_vtag_scores()

    # 计算 VGQA 总分（除“视频内容是否具有美感”外，31 个子属性等权平均）
    vgqa_total = sum(vgqa_scores.values()) / len(vgqa_scores) if vgqa_scores else 0

    # 计算VTAG总分（所有属性的平均分）
    vtag_total = sum(vtag_scores.values()) / len(vtag_scores) if vtag_scores else 0

    print(f"\nVAQA 维度分数: {vaqa_scores}")
    print(f"VGQA 总分（31 个子属性平均）: {vgqa_total:.4f}")
    print(f"VTAG 总分（各属性平均）: {vtag_total:.4f}")

    # 写入汇总CSV
    with open(OUTPUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)

        # 写入标题行
        writer.writerow(['模型', '任务', '维度/属性', '分数'])

        # 写入VAQA各维度
        writer.writerow([MODEL_NAME, 'VAQA', '各维度分数', ''])
        for dim_en, score in vaqa_scores.items():
            writer.writerow([MODEL_NAME, 'VAQA', dim_en, f"{score:.4f}"])

        # 写入VAQA总分（整体）
        if 'overall' in vaqa_scores:
            writer.writerow([MODEL_NAME, 'VAQA', '总分(整体)', f"{vaqa_scores['overall']:.4f}"])

        writer.writerow([])

        # 写入 VGQA 的 31 个 index 问题级子属性
        writer.writerow([MODEL_NAME, 'VGQA', '各子属性分数', ''])
        for attr_name, score in vgqa_scores.items():
            writer.writerow([MODEL_NAME, 'VGQA', attr_name, f"{score:.4f}"])

        # 写入VGQA总分
        writer.writerow([MODEL_NAME, 'VGQA', '总分(平均)', f"{vgqa_total:.4f}"])

        writer.writerow([])

        # 写入VTAG各属性
        writer.writerow([MODEL_NAME, 'VTAG', '各属性准确率', ''])
        for attr, acc in vtag_scores.items():
            writer.writerow([MODEL_NAME, 'VTAG', attr, f"{acc:.4f}"])

        # 写入VTAG总分
        writer.writerow([MODEL_NAME, 'VTAG', '总分(平均)', f"{vtag_total:.4f}"])

    print(f"\n汇总CSV已保存到: {OUTPUT_CSV}")

    # 同时输出一个简洁版本
    simple_csv = os.path.join(OUTPUT_DIR, 'model_scores.csv')
    with open(simple_csv, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['模型', '任务', '总分'])
        writer.writerow([MODEL_NAME, 'VAQA', f"{vaqa_scores.get('overall', 0):.4f}"])
        writer.writerow([MODEL_NAME, 'VGQA', f"{vgqa_total:.4f}"])
        writer.writerow([MODEL_NAME, 'VTAG', f"{vtag_total:.4f}"])

    print(f"简洁版总分CSV已保存到: {simple_csv}")


if __name__ == "__main__":
    main()
