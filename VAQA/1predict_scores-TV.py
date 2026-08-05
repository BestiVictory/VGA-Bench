from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import sys
import os
import csv
import tempfile
from pathlib import Path

import torch
import numpy as np
import random
import argparse
from tqdm import tqdm
from PIL import Image
import cv2

from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_0506 import CLIP4Clip
from dataloaders.rawvideo_util import RawVideoExtractor
import torch.nn as nn

# -------------------------- 配置参数 --------------------------
def get_args():
    parser = argparse.ArgumentParser(description='Unified Video Scoring: Aesthetic + Character Attributes')

    # 核心配置：视频目录 + 输出
    parser.add_argument('--video_root_dir', type=str,
                        default='../test/200-aesthetic-prompts/videos',
                        help='视频根目录（会遍历所有子文件夹）')
    parser.add_argument('--output_csv', type=str,
                        default='../VAQA_output.csv',
                        help='保存合并分数的输出CSV文件')
    parser.add_argument('--temp_feat_dir', type=str,
                        default='../aes_predictall_temp_features',
                        help='临时保存视频特征的目录')

    # 视频格式
    parser.add_argument('--video_extensions', type=str, nargs='+',
                        default=['mp4', 'webm', 'mkv', 'avi', 'gif', 'webp'],
                        help='支持的视频格式')

    # 特征提取参数
    parser.add_argument('--feature_framerate', type=int, default=1)
    parser.add_argument('--max_frames', type=int, default=12)
    parser.add_argument('--max_words', type=int, default=32)
    parser.add_argument('--eval_frame_order', type=int, default=0)
    parser.add_argument('--slice_framepos', type=int, default=2)
    parser.add_argument('--image_resolution', type=int, default=224)

    # CLIP4Clip 模型
    parser.add_argument("--init_model", default="./checkpoints/pytorch_model.bin.0", type=str)
    parser.add_argument("--cross_model", default="cross-base", type=str)
    parser.add_argument('--freeze_layer_num', type=int, default=0)
    parser.add_argument('--linear_patch', type=str, default="2d")
    parser.add_argument('--sim_header', type=str, default="meanP")
    parser.add_argument('--text_num_hidden_layers', type=int, default=12)
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12)
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4)
    parser.add_argument("--pretrained_clip_name", default="ViT-B/32", type=str)
    parser.add_argument('--loose_type', action='store_true', default=True)
    parser.add_argument('--use_mil', action='store_false', default=False)
    parser.add_argument('--sampled_use_mil', action='store_false', default=False)

    # 评分模型路径
    parser.add_argument('--aesthetic_model_path', type=str,
                        default='./checkpoints/通用属性分-model/best_model_epoch45_loss0.7096.pth')
    parser.add_argument('--character_model_path', type=str,
                        default='./checkpoints/人物属性分-model/best_model_epoch48_loss0.6425.pth')

    parser.add_argument('--input_dim', type=int, default=512)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256])

    # 其他
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_pair', type=int, default=1)

    args = parser.parse_args()
    return args

# -------------------------- WEBP 转视频 --------------------------
def webp_to_temp_video(webp_path, fps=1):
    try:
        img = Image.open(webp_path)
        temp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_path = temp.name
        temp.close()

        frames = []
        if getattr(img, 'is_animated', False):
            for i in range(img.n_frames):
                img.seek(i)
                frame = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
                frames.append(frame)
        else:
            frame = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)
            frames = [frame] * 12

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        return temp_path
    except:
        return None

# -------------------------- 视频特征提取 --------------------------
class VideoFeatureExtractor:
    def __init__(self, args):
        self.feature_framerate = args.feature_framerate
        self.max_frames = args.max_frames
        self.frame_order = args.eval_frame_order
        self.slice_framepos = args.slice_framepos
        self.n_pair = args.n_pair
        self.size = args.image_resolution
        self.extractor = RawVideoExtractor(framerate=args.feature_framerate, size=args.image_resolution)

    def extract(self, video_path):
        tmp = None
        try:
            if not os.path.exists(video_path):
                return None

            if video_path.lower().endswith('.webp'):
                tmp = webp_to_temp_video(video_path)
                if not tmp:
                    return None
                video_path = tmp

            data = self.extractor.get_video_data(video_path)['video']
            if len(data.shape) <= 3:
                return None

            sliced = self.extractor.process_raw_data(data)
            if len(sliced.shape) == 5 and sliced.shape[1] == 1:
                sliced = sliced.squeeze(1)

            if self.max_frames < sliced.shape[0]:
                if self.slice_framepos == 0:
                    sliced = sliced[:self.max_frames]
                elif self.slice_framepos == 1:
                    sliced = sliced[-self.max_frames:]
                else:
                    idx = np.linspace(0, sliced.shape[0]-1, self.max_frames, dtype=int)
                    sliced = sliced[idx]

            sliced = self.extractor.process_frame_order(sliced, self.frame_order)
            L = sliced.shape[0]

            video_mask = np.zeros((self.n_pair, self.max_frames), dtype=np.int64)
            video_mask[0, :L] = 1

            video_tensor = np.zeros((1, self.n_pair, 1, self.max_frames, 3, self.size, self.size), dtype=np.float32)
            video_tensor[0,0,0,:L] = sliced

            return video_tensor, video_mask

        except:
            return None
        finally:
            if tmp and os.path.exists(tmp):
                try: os.unlink(tmp)
                except: pass

# -------------------------- 评分模型（完全对齐原始代码） --------------------------
class AestheticPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=1):
        super(AestheticPredictor, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.Mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.Mlp(x)

class IntegratedAestheticPredictor(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[512,256], num_branches=7):
        super().__init__()
        self.branches = nn.ModuleList([
            AestheticPredictor(input_dim, hidden_dims) for _ in range(num_branches)
        ])
    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)

# -------------------------- 模型加载 --------------------------
def load_clip_model(args, device):
    cache = os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'local')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache, state_dict=None, task_config=args)
    if os.path.exists(args.init_model):
        ckpt = torch.load(args.init_model, map_location='cpu', weights_only=True)
        state = model.state_dict()
        filtered = {k:v for k,v in ckpt.items() if k in state and state[k].shape == v.shape}
        state.update(filtered)
        model.load_state_dict(state)
    model.to(device).eval()
    return model

def load_scoring_model(weight_path, model_class, device):
    model = model_class()
    ckpt = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt)
    model.to(device).eval()
    return model

# -------------------------- 遍历视频 --------------------------
def scan_videos(root, exts):
    exts = {e.lower() for e in exts}
    paths = []
    root_path = Path(root)

    # 遍历 一级子目录
    for sub_dir in root_path.iterdir():
        if sub_dir.is_dir():
            # 递归扫描这个子目录下的所有视频
            for p in sub_dir.rglob("*"):
                if p.suffix.lower().lstrip('.') in exts:
                    paths.append(str(p))

    return sorted(list(set(paths)))

# -------------------------- 主函数 --------------------------
def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[Device] {device}')

    # 目录
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    os.makedirs(args.temp_feat_dir, exist_ok=True)

    # 扫描视频
    all_videos = scan_videos(args.video_root_dir, args.video_extensions)
    print(f'[Total] {len(all_videos)}')

    # 分数列
    aesthetic_names = [
        'score_total','composition','shotsize','lighting','visualtone','color','depthoffield'
    ]
    character_names = ['expression','costume','makeup']
    all_cols = ['video_path'] + aesthetic_names + character_names

    if not all_videos:
        with open(args.output_csv, 'w', encoding='utf-8', newline='') as f:
            csv.writer(f).writerow(all_cols)
        print('[WARN] 未发现可处理视频，已生成空结果文件。')
        return

    # 加载模型
    print('Loading CLIP...')
    clip = load_clip_model(args, device)
    print('Loading aesthetic model (7 branches)...')
    aesthetic = load_scoring_model(args.aesthetic_model_path, 
                                  lambda: IntegratedAestheticPredictor(num_branches=7), device)
    print('Loading character model (3 branches)...')
    character = load_scoring_model(args.character_model_path,
                                  lambda: IntegratedAestheticPredictor(num_branches=3), device)

    # 提取器
    feat_ext = VideoFeatureExtractor(args)

    # 一次性执行：创建全新文件，每个成功视频立即落盘。
    success_count = 0
    failed_paths = []
    with open(args.output_csv, 'w', encoding='utf-8', newline='') as output_f:
        writer = csv.writer(output_f)
        writer.writerow(all_cols)
        output_f.flush()

        with torch.no_grad():
            for path in tqdm(all_videos, desc='Scoring'):
                try:
                    feat = feat_ext.extract(path)
                    if feat is None:
                        raise ValueError('视频特征提取失败')

                    v_data, v_mask = feat
                    v_tensor = torch.from_numpy(v_data).to(device, dtype=torch.float32)
                    m_tensor = torch.from_numpy(v_mask).to(device, dtype=torch.int64)

                    vis = clip.get_visual_output(v_tensor, m_tensor)
                    feat_512 = vis.mean(dim=1)
                    scores_aes = aesthetic(feat_512).cpu().numpy().flatten()
                    scores_char = character(feat_512).cpu().numpy().flatten()

                    scores_aes_2d = [f"{s:.2f}" for s in scores_aes]
                    scores_char_2d = [f"{s:.2f}" for s in scores_char]
                    writer.writerow([path] + scores_aes_2d + scores_char_2d)
                    output_f.flush()
                    success_count += 1
                except Exception as exc:
                    failed_paths.append(path)
                    print(f'[ERROR] VAQA 视频评分失败: {path} | {exc}')

    print(f'✅ Done! 成功 {success_count}/{len(all_videos)}，结果保存到 {args.output_csv}')
    if failed_paths:
        print(f'[WARN] VAQA 共有 {len(failed_paths)} 个视频未产生评分。')

if __name__ == '__main__':
    main()
