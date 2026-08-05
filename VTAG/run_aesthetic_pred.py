import json
import os
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ====================== 配置参数 ======================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.environ.get(
    "QWEN_MODEL_PATH", os.path.join(PROJECT_ROOT, "Qwen3-VL-32B-Instruct")
)
LORA_PATH = os.environ.get(
    "VTAG_LORA_PATH", os.path.join(SCRIPT_DIR, "checkpoint-770")
)
PROMPT_PATH = os.environ.get(
    "VTAG_PROMPT_PATH", os.path.join(SCRIPT_DIR, "prompt_Mayi.txt")
)
VIDEO_EXTENSIONS = {'.mp4', '.webm', '.gif', '.avi', '.mov'}

VIDEO_DIR = os.environ.get(
    'VTAG_VIDEO_DIR', os.path.join(PROJECT_ROOT, 'test', '220-tag-prompts', 'videos')
)
OUTPUT_PATH = os.environ.get(
    'VTAG_OUTPUT_PATH', os.path.join(SCRIPT_DIR, 'aesthetic_pred_results.jsonl')
)
MAX_NEW_TOKENS = int(os.environ.get('VTAG_MAX_NEW_TOKENS', '2048'))
TEMPERATURE = float(os.environ.get('VTAG_TEMPERATURE', '0.001'))
TOP_P = float(os.environ.get('VTAG_TOP_P', '1.0'))

# ====================== 加载 Prompt ======================
def load_prompt(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

# ====================== 加载模型和 Processor ======================
def load_model_and_processor(model_path, lora_path):
    print("加载 processor...")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        fps=1,
        max_frame_num=6
    )

    print("加载基础模型...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print("加载并合并 LoRA 权重...")
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model.eval()

    return model, processor

# ====================== 清理模型输出 ======================
def clean_response(response):
    dual_choice_fields = ["画面结构", "光源位置", "光的颜色", "景别类型", "色相"]
    single_choice_fields = ["光源数量", "光的质量", "景深范围", "饱和度", "亮度", "色温", "对比度"]
    all_required_fields = dual_choice_fields + single_choice_fields + ["置信度"]

    valid_values = {
        "画面结构": ["对称构图", "非对称构图", "三分法构图", "中心构图", "框架构图", "引导线构图", "其他"],
        "光源数量": ["单光源", "多光源"],
        "光源位置": ["顺光", "侧光", "逆光", "顶光", "底光"],
        "光的质量": ["软光", "硬光"],
        "光的颜色": ["白（中性）光", "暖光", "冷光", "彩色光"],
        "景别类型": ["远景", "全景", "中景", "近景", "特写"],
        "景深范围": ["浅景深", "深景深"],
        "饱和度": ["高", "中", "低"],
        "亮度": ["亮", "中", "暗"],
        "色相": ["红", "橙", "黄", "绿", "蓝", "紫", "其他"],
        "色温": ["冷", "中", "暖"],
        "对比度": ["高", "中", "低"]
    }

    try:
        import re
        response = response.strip()

        # 1) 尝试匹配完整的 JSON 数组
        json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            json_str = json_str.replace('\n', ' ').replace('\r', '').replace('\t', '')
            result = json.loads(json_str)
        else:
            # 2) JSON 被截断：用大括号配对提取完整的 {...} 对象
            result = []
            i = 0
            while i < len(response):
                if response[i] == '{':
                    depth = 0
                    start = i
                    while i < len(response):
                        if response[i] == '{':
                            depth += 1
                        elif response[i] == '}':
                            depth -= 1
                            if depth == 0:
                                obj_str = response[start:i+1]
                                try:
                                    parsed = json.loads(obj_str)
                                    if isinstance(parsed, dict) and len(parsed) >= 5:
                                        result.append(parsed)
                                except json.JSONDecodeError:
                                    pass
                                break
                        i += 1
                i += 1
            if not result:
                raise ValueError("未找到可解析的JSON对象")

        if isinstance(result, list):
            validated_result = []
            for frame in result:
                if not isinstance(frame, dict):
                    continue
                validated_frame = {}
                for field in all_required_fields:
                    if field == "置信度":
                        val = float(frame.get(field, 0.0))
                        validated_frame[field] = max(0.0, min(1.0, val))
                    else:
                        val = frame.get(field, {}).get("value", "")
                        validated_frame[field] = {"value": val}
                validated_result.append(validated_frame)
            return validated_result

    except Exception as e:
        print(f"解析失败: {str(e)[:100]}")

    return [{
        "画面结构": {"value": ""}, "光源数量": {"value": ""}, "光源位置": {"value": ""},
        "光的质量": {"value": ""}, "光的颜色": {"value": ""}, "景别类型": {"value": ""},
        "景深范围": {"value": ""}, "饱和度": {"value": ""}, "亮度": {"value": ""},
        "色相": {"value": ""}, "色温": {"value": ""}, "对比度": {"value": ""},
        "置信度": 0.0, "error": "解析失败"
    }]

# ====================== 处理单个视频 ======================
def process_video(video_path, model, processor, prompt_content):
    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt_content}
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
                use_cache=True
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0]

        return clean_response(response)

    except Exception as e:
        print(f"处理视频失败: {str(e)}")
        return [{
            "画面结构": {"value": ""}, "光源数量": {"value": ""}, "光源位置": {"value": ""},
            "光的质量": {"value": ""}, "光的颜色": {"value": ""}, "景别类型": {"value": ""},
            "景深范围": {"value": ""}, "饱和度": {"value": ""}, "亮度": {"value": ""},
            "色相": {"value": ""}, "色温": {"value": ""}, "对比度": {"value": ""},
            "置信度": 0.0, "error": "处理失败"
        }]

# ====================== 扫描视频目录 ======================
def scan_video_dir(video_dir):
    video_files = []
    for fname in sorted(os.listdir(video_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            video_files.append(os.path.join(video_dir, fname))
    return video_files

# ====================== 主函数 ======================
def main():
    prompt_content = load_prompt(PROMPT_PATH)
    model, processor = load_model_and_processor(MODEL_PATH, LORA_PATH)

    video_files = scan_video_dir(VIDEO_DIR)
    print(f"共发现 {len(video_files)} 个视频文件")

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_PATH)), exist_ok=True)
    print(f"本次一次性处理 {len(video_files)} 个视频")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as fout:
        for video_path in tqdm(video_files, desc="推理进度"):
            annotation_result = process_video(video_path, model, processor, prompt_content)
            final_annotation = [annotation_result[0]] if isinstance(annotation_result, list) else []

            result = {
                "video_path": video_path,
                "annotation": final_annotation
            }
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            fout.flush()

    print(f"推理完成，结果已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
