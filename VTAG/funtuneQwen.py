import json
import os
import sys
import torch
import random
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
JSON_PATH = os.environ.get(
    "VTAG_INPUT_JSON", os.path.join(SCRIPT_DIR, "input.json")
)
PROMPT_PATH = os.environ.get(
    "VTAG_PROMPT_PATH", os.path.join(SCRIPT_DIR, "prompt_Mayi.txt")
)
OUTPUT_PATH = os.environ.get(
    "VTAG_OUTPUT_JSON", os.path.join(SCRIPT_DIR, "video_annotation_results.json")
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.001
TOP_P = 1.0
SAMPLE_SIZE = 1000
RANDOM_SEED = 42

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
        #attn_implementation="flash_attention_2",
    )

    print("加载并合并 LoRA 权重...")
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()
    model.eval()

    return model, processor

# ====================== 【修复】清理模型输出 ======================
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
        # ====================== 【关键修复】直接提取JSON ======================
        import re
        response = response.strip()
        # 正则匹配最外层 [] 内容
        json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
        if not json_match:
            raise ValueError("未找到JSON数组")
        
        json_str = json_match.group(0)
        json_str = json_str.replace('\n', ' ').replace('\r', '').replace('\t', '')
        result = json.loads(json_str)

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
        print(f"修复版解析失败: {str(e)[:100]}")

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
                do_sample=False,  # 【关键加速】关闭采样，贪婪推理
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
            "置信度": 0.0, "error": f"处理失败"
        }]

# ====================== 读取已处理视频 ======================
def get_processed_videos(jsonl_path):
    processed_videos = set()
    if not os.path.exists(jsonl_path):
        return processed_videos
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                if "video_path" in item:
                    processed_videos.add(item["video_path"])
    except:
        pass
    return processed_videos

# ====================== 主函数 ======================
def main():
    prompt_content = load_prompt(PROMPT_PATH)
    model, processor = load_model_and_processor(MODEL_PATH, LORA_PATH)

    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    random.seed(RANDOM_SEED)
    sampled_data = random.sample(test_data, min(SAMPLE_SIZE, len(test_data)))

    output_dir = os.path.dirname(OUTPUT_PATH)
    os.makedirs(output_dir, exist_ok=True)
    OUTPUT_JSONL = OUTPUT_PATH.replace('.json', '.jsonl')
    processed_videos = get_processed_videos(OUTPUT_JSONL)

    total_to_process = sum(1 for item in sampled_data if item.get("videos") and os.path.exists(item["videos"][0]) and item["videos"][0] not in processed_videos)
    pbar = tqdm(total=total_to_process, desc="推理进度")

    with open(OUTPUT_JSONL, 'a', encoding='utf-8') as fout:
        for item in sampled_data:
            video_paths = item.get("videos", [])
            if not video_paths:
                continue
            video_path = video_paths[0]

            if not os.path.exists(video_path) or video_path in processed_videos:
                pbar.update(1)
                continue

            annotation_result = process_video(video_path, model, processor, prompt_content)
            final_annotation = [annotation_result[0]] if isinstance(annotation_result, list) else []

            result = {
                "video_path": video_path,
                "url": item.get("url", []),
                "annotation": final_annotation
            }
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')
            fout.flush()
            pbar.update(1)

    pbar.close()
    print(f"✅ 推理完成！结果已保存至: {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
