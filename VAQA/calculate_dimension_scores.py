#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算每个维度的平均分（根据prompt对应的维度进行筛选）
"""

import csv
import os
from collections import defaultdict

# 文件路径（相对于项目根目录，与 run_all.sh 的配置保持一致）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VAQA_PRED_PATH = os.environ.get(
    "VAQA_PRED_PATH", os.path.join(PROJECT_ROOT, "vaqa_pred_results.csv")
)
TEST_PROMPT_PATH = os.environ.get(
    "VAQA_TEST_PROMPT_PATH",
    os.path.join(PROJECT_ROOT, "test", "200-aesthetic-prompts", "200-aesthetic-prompts.csv"),
)
CVPR_PROMPT_PATH = os.environ.get(
    "VAQA_GT_PROMPT_PATH",
    os.path.join(PROJECT_ROOT, "prompts", "200-aesthetic-prompts.csv"),
)
VAQA_DIM_OUTPUT_PATH = os.environ.get(
    "VAQA_DIM_OUTPUT_PATH", os.path.join(SCRIPT_DIR, "vaqa_dim_scores.csv")
)

# 维度列名（英文）
DIMENSION_COLUMNS = ['composition', 'shotsize', 'lighting', 'visualtone', 'color', 'depthoffield', 'expression', 'costume', 'makeup']

# 维度中英文映射
DIMENSION_CN_TO_EN = {
    '整体': 'overall',
    '构图': 'composition',
    '景别': 'shotsize',
    '光照': 'lighting',
    '影调': 'visualtone',
    '色彩': 'color',
    '景深': 'depthoffield',
    '表情': 'expression',
    '服装': 'costume',
    '化妆': 'makeup'
}

# 英文维度到CSV列名的映射
DIMENSION_EN_TO_COL = {
    'overall': 'score_total',  # 整体对应总分
    'composition': 'composition',
    'shotsize': 'shotsize',
    'lighting': 'lighting',
    'visualtone': 'visualtone',
    'color': 'color',
    'depthoffield': 'depthoffield',
    'expression': 'expression',
    'costume': 'costume',
    'makeup': 'makeup'
}


def load_video_prompt_mapping():
    """加载视频名称到prompt的映射"""
    video_to_prompt = {}
    with open(TEST_PROMPT_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name'].strip()
            en_prompt = row['en_prompt'].strip().strip('"')
            video_to_prompt[name] = en_prompt
    print(f"加载了 {len(video_to_prompt)} 个视频-prompt映射")
    return video_to_prompt


def load_prompt_dimension_mapping():
    """加载prompt到维度的映射"""
    prompt_to_dimensions = {}
    with open(CVPR_PROMPT_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            # 获取英文prompt
            en_prompt = row.get('en_prompt', '').strip().strip('"')
            dimension_cn = row.get('各组合维度', '').strip()

            if en_prompt and dimension_cn:
                # 解析维度组合
                dimensions = [d.strip() for d in dimension_cn.split(' + ')]
                prompt_to_dimensions[en_prompt] = dimensions

    print(f"加载了 {len(prompt_to_dimensions)} 个prompt-维度映射")
    return prompt_to_dimensions


def extract_video_name(video_path):
    """从video_path提取视频文件名"""
    # 例如: test/200-aesthetic-prompts/videos/009200ff077ca6babc6611ccd5eca836.webm
    basename = os.path.basename(video_path)
    return basename


def find_matching_dimensions(prompt, prompt_to_dimensions):
    """查找匹配的维度"""
    # 精确匹配
    if prompt in prompt_to_dimensions:
        return prompt_to_dimensions[prompt]

    # 尝试部分匹配
    for key, dims in prompt_to_dimensions.items():
        if key in prompt or prompt in key:
            return dims

    return None


def translate_dimensions_to_english(dimensions_cn):
    """将中文维度翻译成英文维度列表"""
    dimensions_en = []
    for dim in dimensions_cn:
        if dim in DIMENSION_CN_TO_EN:
            dimensions_en.append(DIMENSION_CN_TO_EN[dim])
    return dimensions_en


def main():
    # 加载数据
    print("=" * 60)
    print("步骤1: 加载视频-prompt映射")
    print("=" * 60)
    video_to_prompt = load_video_prompt_mapping()

    print("\n" + "=" * 60)
    print("步骤2: 加载prompt-维度映射")
    print("=" * 60)
    prompt_to_dimensions = load_prompt_dimension_mapping()

    print("\n" + "=" * 60)
    print("步骤3: 处理VAQA预测结果")
    print("=" * 60)

    # 存储每个维度的分数
    dimension_scores = defaultdict(list)

    # 统计信息
    matched_count = 0
    unmatched_count = 0
    unmatched_prompts = []

    with open(VAQA_PRED_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_path = row['video_path'].strip()
            video_name = extract_video_name(video_path)

            # 获取对应的prompt
            if video_name not in video_to_prompt:
                unmatched_count += 1
                continue

            prompt = video_to_prompt[video_name]

            # 获取对应的维度
            dimensions_cn = find_matching_dimensions(prompt, prompt_to_dimensions)

            if dimensions_cn is None:
                unmatched_count += 1
                if prompt not in unmatched_prompts:
                    unmatched_prompts.append(prompt)
                continue

            matched_count += 1

            # 翻译维度为英文
            dimensions_en = translate_dimensions_to_english(dimensions_cn)

            # 获取该视频各维度的分数
            for dim_en in dimensions_en:
                if dim_en in DIMENSION_EN_TO_COL:
                    col_name = DIMENSION_EN_TO_COL[dim_en]
                    score = float(row.get(col_name, 0))
                    dimension_scores[dim_en].append(score)

    print(f"成功匹配: {matched_count} 个视频")
    print(f"未匹配: {unmatched_count} 个视频")

    if unmatched_prompts:
        print(f"\n未匹配的prompt示例 (前5个):")
        for p in unmatched_prompts[:5]:
            print(f"  - {p[:80]}...")

    print("\n" + "=" * 60)
    print("步骤4: 计算每个维度的平均分")
    print("=" * 60)

    # 计算并输出每个维度的平均分
    results = []
    for dim_en in ['overall', 'composition', 'shotsize', 'lighting', 'visualtone', 'color', 'depthoffield', 'expression', 'costume', 'makeup']:
        if dimension_scores[dim_en]:
            avg_score = sum(dimension_scores[dim_en]) / len(dimension_scores[dim_en])
            avg_score = avg_score / 10.0  # 除以10归一化到0-1范围
            count = len(dimension_scores[dim_en])
            results.append((dim_en, avg_score, count))

    # 按平均分排序输出
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print("各维度平均分统计（按分数降序排列，除以10归一化）")
    print("=" * 60)
    print(f"{'排名':<5} {'中文维度':<12} {'英文维度':<18} {'平均分':<12} {'样本数':<10}")
    print("-" * 60)

    for i, (dim_en, avg_score, count) in enumerate(results, 1):
        dim_cn = [k for k, v in DIMENSION_CN_TO_EN.items() if v == dim_en]
        dim_cn_display = dim_cn[0] if dim_cn else "-"
        print(f"{i:<5} {dim_cn_display:<12} {dim_en:<18} {avg_score:<12.4f} {count:<10}")

    # 保存CSV（保存到当前脚本所在目录）
    csv_path = VAQA_DIM_OUTPUT_PATH
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, 'w', encoding='utf-8-sig') as f:
        f.write('维度(英文),维度(中文),平均分,样本数\n')
        for dim_en, avg_score, count in results:
            dim_cn = [k for k, v in DIMENSION_CN_TO_EN.items() if v == dim_en]
            dim_cn_display = dim_cn[0] if dim_cn else "-"
            f.write(f"{dim_en},{dim_cn_display},{avg_score:.4f},{count}\n")
    print(f"\nCSV 已保存到: {csv_path}")

    # 输出汇总
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"总共处理视频数: {matched_count + unmatched_count}")
    print(f"成功匹配维度: {matched_count}")
    print(f"未匹配: {unmatched_count}")


if __name__ == "__main__":
    main()
