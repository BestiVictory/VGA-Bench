import json
import csv
import os
from collections import defaultdict

# 文件路径（相对于项目根目录，与 run_all.sh 的配置保持一致）
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
test_csv = os.environ.get(
    "VTAG_TEST_CSV",
    os.path.join(project_root, "test", "220-tag-prompts", "220-tag-prompts.csv"),
)

# === 1. 读取预测结果 ===
pred_file = os.environ.get(
    "VTAG_PRED_FILE", os.path.join(project_root, "vtag_pred_results.jsonl")
)
predictions = {}  # video_filename -> {attr: set_of_values}

with open(pred_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        video_path = data["video_path"]
        video_filename = os.path.basename(video_path)
        ann = data["annotation"][0]
        pred_attrs = {}
        for attr, val_dict in ann.items():
            if attr == "置信度":
                continue
            val = val_dict.get("value", "")
            if val:
                pred_attrs[attr] = set(v.strip() for v in val.split(","))
            else:
                pred_attrs[attr] = set()
        predictions[video_filename] = pred_attrs

print(f"预测结果: {len(predictions)} 个视频")

# === 2. 读取视频-prompt映射 ===
video_to_prompt = {}  # video_filename -> en_prompt

with open(test_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        video_to_prompt[row["name"]] = row["en_prompt"]

print(f"视频-prompt映射: {len(video_to_prompt)} 条")

# === 3. 读取真值 ===
gt_csv = os.environ.get(
    "VTAG_GT_CSV", os.path.join(project_root, "prompts", "220-tag-prompts.csv")
)
prompt_to_gt = {}  # en_prompt -> {attr: value}

with open(gt_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        en_prompt = row["en_prompt"]
        gt_attrs = {}
        # 属性名映射: 真值CSV列名 -> 预测JSON属性名
        attr_mapping = {
            "构图": "画面结构",
            "光源数量": "光源数量",
            "光源位置": "光源位置",
            "光的质量": "光的质量",
            "光的色彩": "光的颜色",
            "景别类型": "景别类型",
            "景深范围": "景深范围",
            "饱和度": "饱和度",
            "亮度": "亮度",
            "色温": "色温",
            "对比度": "对比度",
        }
        for gt_attr, pred_attr in attr_mapping.items():
            val = row.get(gt_attr, "").strip()
            if val:
                gt_attrs[pred_attr] = val  # 用预测属性名作为key
        # 同一英文 prompt 出现多次时保留第一条；当前数据即让 T013 覆盖 T014。
        if en_prompt not in prompt_to_gt:
            prompt_to_gt[en_prompt] = gt_attrs

print(f"真值: {len(prompt_to_gt)} 条prompt")

# === 4. 值映射（处理真值和预测值名称不一致的情况） ===
value_mapping = {
    "白光": "白（中性）光",
}

def normalize_value(val):
    return value_mapping.get(val, val)

# === 5. 计算准确率 ===
# 对每个属性，统计：总数（真值非空）、正确数
attr_stats = defaultdict(lambda: {"total": 0, "correct": 0})

matched_videos = 0
unmatched_videos = 0

for video_filename, pred_attrs in predictions.items():
    # 找到对应的prompt
    en_prompt = video_to_prompt.get(video_filename)
    if not en_prompt:
        unmatched_videos += 1
        continue

    # 找到对应的真值
    gt_attrs = prompt_to_gt.get(en_prompt)
    if not gt_attrs:
        unmatched_videos += 1
        continue

    matched_videos += 1

    # 对每个有真值的属性进行对比
    for attr, gt_val in gt_attrs.items():
        if not gt_val:
            continue
        attr_stats[attr]["total"] += 1

        # 真值标准化
        normalized_gt = normalize_value(gt_val)
        gt_set = set(v.strip() for v in normalized_gt.split(","))

        # 预测值
        pred_set = pred_attrs.get(attr, set())

        # 判断是否正确：真值的所有标签都在预测中出现
        if gt_set.issubset(pred_set):
            attr_stats[attr]["correct"] += 1

print(f"\n匹配视频数: {matched_videos}, 未匹配: {unmatched_videos}")

# === 6. 输出结果 ===
print("\n" + "=" * 60)
print(f"{'属性':<12} {'正确/总数':<15} {'准确率':<10}")
print("=" * 60)

total_correct = 0
total_count = 0

# 按照固定顺序输出
attr_order = ["画面结构", "光源位置", "光的颜色", "景别类型", "光源数量", "光的质量", "景深范围", "饱和度", "亮度", "色温", "对比度"]

for attr in attr_order:
    if attr in attr_stats:
        stats = attr_stats[attr]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"{attr:<12} {stats['correct']}/{stats['total']:<12} {acc:.2f}%")
        total_correct += stats["correct"]
        total_count += stats["total"]

print("=" * 60)
overall_acc = total_correct / total_count * 100 if total_count > 0 else 0
print(f"{'总体':<12} {total_correct}/{total_count:<12} {overall_acc:.2f}%")

# 保存CSV
csv_path = os.environ.get(
    "VTAG_SCORE_OUTPUT_PATH", os.path.join(script_dir, 'vtag_dim_scores.csv')
)
os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
with open(csv_path, 'w', encoding='utf-8-sig') as f:
    f.write('属性,正确数,总数,准确率\n')
    for attr in attr_order:
        if attr in attr_stats:
            stats = attr_stats[attr]
            acc = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            f.write(f"{attr},{stats['correct']},{stats['total']},{acc:.4f}\n")
    f.write(f"总体,{total_correct},{total_count},{overall_acc:.4f}\n")
print(f"\nCSV 已保存到: {csv_path}")
