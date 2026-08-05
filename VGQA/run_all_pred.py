import os
import json
import sys
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from PIL import Image

from swift import get_model_processor, get_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
from peft import PeftModel

# ======================== 配置区 ========================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_PATH = os.environ.get(
    'QWEN_MODEL_PATH', str(PROJECT_ROOT / 'Qwen3-VL-32B-Instruct')
)
LORA_CHECKPOINT_DIR = os.environ.get(
    'VGQA_LORA_PATH', str(SCRIPT_DIR / 'v7-20260112-204659' / 'checkpoint-570')
)
TEMPLATE_TYPE = None
DEFAULT_SYSTEM = None
DEFAULT_MODEL_NAME = os.environ.get('MODEL_NAME', 'default')
PROMPT_FILE = os.environ.get('VGQA_PROMPT_FILE', str(SCRIPT_DIR / 'prompt_gqa_mayi.txt'))

OUTPUT_FILE = os.environ.get('VGQA_OUTPUT_FILE', 'all_pred_results.jsonl')
TEMP_DIR = os.environ.get(
    'VGQA_TEMP_DIR', os.path.join(os.path.dirname(os.path.abspath(OUTPUT_FILE)), 'vgqa_temp')
)
BATCH_SIZE = int(os.environ.get('VGQA_BATCH_SIZE', '2'))
MAX_TOKENS = int(os.environ.get('VGQA_MAX_TOKENS', '512'))
TEMPERATURE = int(os.environ.get('VGQA_TEMPERATURE', '0'))
QO_CSV = os.environ.get('VGQA_QO_CSV', str(SCRIPT_DIR / 'prompt_qo_deduped.csv'))

_env_csvs = os.environ.get('VGQA_DATA_CSVS', '')
_env_dirs = os.environ.get('VGQA_DATA_VIDEO_DIRS', '')
if _env_csvs and _env_dirs:
    DATA_SOURCES = [
        {'csv': c.strip(), 'video_dir': d.strip()}
        for c, d in zip(_env_csvs.split(','), _env_dirs.split(','))
    ]
else:
    DATA_SOURCES = [
        {
            'csv': str(PROJECT_ROOT / 'test' / '120-base_prompts' / '120-base_prompts.csv'),
            'video_dir': str(PROJECT_ROOT / 'test' / '120-base_prompts' / 'videos'),
        },
        {
            'csv': str(PROJECT_ROOT / 'test' / '476-GQprompts' / '476-GQprompts.csv'),
            'video_dir': str(PROJECT_ROOT / 'test' / '476-GQprompts' / 'videos'),
        },
    ]
# ========================================================


def append_to_jsonl(data, filename):
    with open(filename, 'a', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')


def webp2gif(path, gif_path):
    img = Image.open(path)
    img.save(gif_path, 'GIF', save_all=True, disposal=2)
    return gif_path


def load_system_prompt():
    try:
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f'错误：未找到提示词文件 "{PROMPT_FILE}"')
        sys.exit(1)


def build_user_content(prompt, qo_row):
    """参照 create_data.py 的逻辑构建 user 消息内容"""
    qos = [['q1','o1'],['q2','o2'],['q3','o3'],['q4','o4'],['q5','o5'],['q6','o6']]
    line = f'该视频的Prompt是:{prompt}\n分析视频，回答以下问题\n'
    qss = ''
    for q_key, o_key in qos:
        q_val = qo_row.get(q_key)
        o_val = qo_row.get(o_key)
        if pd.notna(q_val) and pd.notna(o_val):
            o_list = str(o_val).split(',')
            line += f'**问题**\n\n"{q_key}":\n{q_val}\n"option":\n'
            for opt in o_list:
                line += f'{opt}\n'
            line += '\n'
            qss += f"{q_key}、"
    if qss:
        line += f'需要回答的有{qss[:-1]}\n'
    return '<video>' + line


def build_infer_data(qo_df, system_prompt):
    """构建所有待推理的数据列表"""
    # 建立 prompt -> q/o 行的映射
    qo_map = {}
    for _, row in qo_df.iterrows():
        qo_map[row['prompt']] = row.to_dict()

    infer_data = []
    skipped = 0

    for source in DATA_SOURCES:
        csv_path = source['csv']
        video_dir = source['video_dir']
        df = pd.read_csv(csv_path)

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f'构建数据 {os.path.basename(csv_path)}'):
            en_prompt = row['en_prompt']
            video_name = row['name']
            video_path = os.path.join(video_dir, video_name)
            source_video_path = video_path

            if not os.path.exists(video_path):
                skipped += 1
                continue

            # webp 转 gif
            if video_path.lower().endswith('.webp'):
                os.makedirs(TEMP_DIR, exist_ok=True)
                source_group = os.path.basename(os.path.dirname(video_dir))
                gif_name = f"{source_group}_{os.path.splitext(video_name)[0]}.gif"
                gif_path = os.path.join(TEMP_DIR, gif_name)
                if not os.path.exists(gif_path):
                    webp2gif(video_path, gif_path)
                video_path = gif_path

            # 查找对应的 q/o 数据
            qo_row = qo_map.get(en_prompt)
            if qo_row is None:
                skipped += 1
                continue

            user_content = build_user_content(en_prompt, qo_row)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            infer_data.append({
                "prompt": en_prompt,
                "model": DEFAULT_MODEL_NAME,
                "messages": messages,
                "videos": [video_path],
                "source_video": source_video_path,
            })

    print(f'共构建 {len(infer_data)} 条推理数据，跳过 {skipped} 条')
    return infer_data


def run_inference(infer_data, engine, request_config, output_file):
    """一次性执行推理并保存到全新的结果文件。"""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8'):
        pass

    for item in tqdm(infer_data, desc='推理中'):
        video_path = item['videos'][0]
        prompt = item['prompt']
        source_video = item['source_video']

        try:
            infer_requests = [
                InferRequest(messages=item['messages'], videos=item['videos'])
            ]
            resp_list = engine.infer(infer_requests, request_config)
            output_text = resp_list[0].choices[0].message.content

            try:
                description_list = json.loads(output_text)
                result = {
                    "prompt": prompt,
                    "model": item['model'],
                    "video": video_path,
                    "source_video": source_video,
                    "gpt": description_list,
                }
            except json.JSONDecodeError:
                result = {
                    "prompt": prompt,
                    "model": item['model'],
                    "video": video_path,
                    "source_video": source_video,
                    "error": "Invalid JSON output",
                    "raw_output": output_text,
                }
            append_to_jsonl(result, output_file)

        except Exception as e:
            print(f"  -> 处理视频 '{video_path}' 时发生错误: {e}")
            result = {
                "prompt": prompt,
                "model": item['model'],
                "video": video_path,
                "source_video": source_video,
                "error": str(e),
            }
            append_to_jsonl(result, output_file)


def main():
    # 1. 加载 system prompt
    system_prompt = load_system_prompt()
    print(f'已加载 system prompt: {system_prompt[:50]}...')

    # 2. 加载 q/o 数据
    qo_df = pd.read_csv(QO_CSV)
    print(f'已加载 {QO_CSV}: {len(qo_df)} 条 prompt-qo 记录')

    # 3. 构建推理数据
    infer_data = build_infer_data(qo_df, system_prompt)

    # 4. 加载模型
    model, processor = get_model_processor(MODEL_PATH)
    if LORA_CHECKPOINT_DIR is not None:
        model = PeftModel.from_pretrained(model, LORA_CHECKPOINT_DIR)
    template = get_template(processor)
    engine = TransformersEngine(model, template=template)
    request_config = RequestConfig(max_tokens=MAX_TOKENS, temperature=TEMPERATURE)

    # 5. 执行推理
    run_inference(infer_data, engine, request_config, OUTPUT_FILE)
    print(f'\n推理完成，结果已保存到: {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
