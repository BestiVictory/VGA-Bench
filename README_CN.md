# VGA-Bench 与 VGA-BenchV2：视频美学与生成质量评测

[English](README.md) | **中文**

本仓库包含以下两项相关工作的视频生成评测组件：

- **VGA-Bench: A Unified Benchmark and Multi-Model Framework for Video Aesthetics and Generation Quality Evaluation**；
- **VGA-BenchV2: An Expanded Unified Benchmark and Multi-Model Framework for Evaluating Video Aesthetics and Generation Quality**（尚未见刊）。

参评者使用仓库中的四套 prompt suite 生成视频，按照规定目录准备视频和 CSV 映射文件后，运行一个脚本即可完成输入检查、三类评测、分数汇总和可选的邮件通知。

> 本仓库只负责评测，不包含待测视频生成模型。参评者需要使用自己的模型生成视频，并自行准备 `test/` 目录中的视频与 CSV。

## 1. VGA-Bench 与 VGA-BenchV2 简介

VGA-Bench 面向文本到视频生成模型，目标是从互补角度评估生成视频，而不是只依赖单一总体指标。VGA-BenchV2 在此基础上进一步扩展了面向视频美学与生成质量的统一基准和多模型评测框架。本仓库包含两项工作所使用的评测组件，当前公开代码包含三条评测链路：

| 任务 | 输入 prompt suite | 主要评价内容 | 输出 |
| --- | --- | --- | --- |
| VAQA | `200-aesthetic-prompts` | 整体审美、构图、景别、光照、影调、色彩、景深、表情、服装、化妆 | 各维度归一化分数与 VAQA 总分 |
| VGQA | `120-base_prompts`、`476-GQprompts` | 通过问题级视频质量问答评估基础质量、时序稳定性、运动、场景和通用生成质量 | 31 个子属性分数与 VGQA 总分 |
| VTAG | `220-tag-prompts` | 构图、光源、景别、景深、饱和度、亮度、色温、对比度等可控视觉属性与 prompt 的一致性 | 11 个属性准确率与 VTAG 总分 |

VGA-Bench 与 VGA-BenchV2 的共同思路是将审美质量、通用视频质量和可控视觉属性一致性分别建模，再以可解释的维度级结果呈现模型能力。当前仓库对应两项工作的评测组件，不包含论文训练代码和视频生成代码。

VGA-Bench 已发表于 IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR）2026；VGA-BenchV2 尚未见刊。因此，Citation 部分继续保留已发表 VGA-Bench 论文的正式引用。

## 2. 仓库结构

```text
.
├── prompts/                         # 官方 prompt suite 与真值
│   ├── 120-base_prompts.csv
│   ├── 200-aesthetic-prompts.csv
│   ├── 220-tag-prompts.csv
│   └── 476-GQprompts.csv
├── test/                            # 由参评者准备的测试视频和映射 CSV
│   ├── 120-base_prompts/
│   │   ├── videos/
│   │   └── 120-base_prompts.csv
│   ├── 200-aesthetic-prompts/
│   │   ├── videos/
│   │   └── 200-aesthetic-prompts.csv
│   ├── 220-tag-prompts/
│   │   ├── videos/
│   │   └── 220-tag-prompts.csv
│   └── 476-GQprompts/
│       ├── videos/
│       └── 476-GQprompts.csv
├── VAQA/                            # 审美与人物属性评分
├── VGQA/                            # 通用视频质量问答评分
├── VTAG/                            # 视觉标签一致性评分
├── validate_run.py                 # 推理前输入检查
├── summarize_scores.py             # 三项任务分数汇总
├── send_email.py                   # 可选的结果邮件发送
├── run_all.sh                      # 一键评测入口
├── requirements.txt
└── environment.yml
```

## 3. 系统与硬件要求

推荐环境：

- Linux 或 WSL2；
- Python 3.10；
- NVIDIA GPU 与 CUDA 12.4 兼容驱动；
- FFmpeg 和 FFprobe；
- 足够的磁盘空间存放 Qwen3-VL-32B、评测权重、5,080 个测试视频及运行输出。

VGQA 和 VTAG 使用 Qwen3-VL-32B-Instruct。32B BF16 模型显存需求很高，实际需求受模型版本、视频帧数和推理框架影响。建议使用大显存 GPU；如使用多张 GPU，可通过 `GPU_ID=0,1,...` 暴露多张设备，并确认当前推理框架能够正确分配模型。

## 4. 安装环境

### 4.1 使用 Conda（推荐）

在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate cvpr-video-eval
```

`environment.yml` 会安装 Python 3.10、FFmpeg，并调用根目录的 `requirements.txt` 安装推理依赖。

### 4.2 使用 pip

如果系统已经安装兼容的 FFmpeg、Python 3.10 和 CUDA，可执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

检查 FFmpeg：

```bash
ffmpeg -version
ffprobe -version
```

## 5. 下载基础模型与评测权重

评测所需模型和权重已上传至 Google Drive：

**[下载 VGA-Bench 与 VGA-BenchV2 评测模型和权重](https://drive.google.com/drive/folders/18j3ExC10LgmnE47k70YvATtXNbhnogMW?usp=drive_link)**

下载后按以下结构放置。不要只下载 `.safetensors` 而遗漏对应的 `adapter_config.json`。

```text
.
├── Qwen3-VL-32B-Instruct/
├── VAQA/
│   ├── checkpoints/
│   │   ├── pytorch_model.bin.0
│   │   ├── 通用属性分-model/
│   │   │   └── best_model_epoch45_loss0.7096.pth
│   │   └── 人物属性分-model/
│   │       └── best_model_epoch48_loss0.6425.pth
│   └── modules/
│       └── ViT-B-32.pt
├── VGQA/
│   └── v7-20260112-204659/
│       └── checkpoint-570/
│           ├── adapter_config.json
│           └── adapter_model.safetensors
└── VTAG/
    └── checkpoint-770/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

如果模型没有放在默认位置，可以通过环境变量指定：

```bash
export QWEN_MODEL_PATH=/absolute/path/to/Qwen3-VL-32B-Instruct
export VGQA_LORA_PATH=/absolute/path/to/VGQA/checkpoint-570
export VTAG_LORA_PATH=/absolute/path/to/VTAG/checkpoint-770
```

## 6. 准备参评视频

### 6.1 每个 prompt 生成 5 个视频

参评者必须对 prompt suite 中的每个 prompt 生成 **5 个视频**。5 个视频可以对应不同随机种子，但应使用同一个待测视频生成模型和一致的生成设置。

四套 prompt suite 的要求如下：

| Prompt suite | Prompt 数 | 每个 prompt 视频数 | 应准备视频数 |
| --- | ---: | ---: | ---: |
| `120-base_prompts` | 120 | 5 | 600 |
| `200-aesthetic-prompts` | 200 | 5 | 1,000 |
| `220-tag-prompts` | 220 | 5 | 1,100 |
| `476-GQprompts` | 476 | 5 | 2,380 |
| **合计** | 1,016 | 5 | **5,080** |

评测映射以英文 prompt 文本为键。不要自行改写、翻译、增删标点或改变大小写；请从官方 prompt suite 原样复制。

### 6.2 视频目录

将生成视频分别放入：

```text
test/120-base_prompts/videos/
test/200-aesthetic-prompts/videos/
test/220-tag-prompts/videos/
test/476-GQprompts/videos/
```

同一文件夹中的文件名必须唯一。推荐命名方式：

```text
B0001_seed01.mp4
B0001_seed02.mp4
B0001_seed03.mp4
B0001_seed04.mp4
B0001_seed05.mp4
```

文件名不需要与 prompt ID 完全相同，但必须与对应测试 CSV 的 `name` 字段完全一致。

### 6.3 测试 CSV 格式

每个视频必须在对应测试 CSV 中占一行。四份测试 CSV 均使用相同的两列格式：

```csv
name,en_prompt
B0001_seed01.mp4,"An example English prompt copied exactly from the prompt suite."
B0001_seed02.mp4,"An example English prompt copied exactly from the prompt suite."
B0001_seed03.mp4,"An example English prompt copied exactly from the prompt suite."
B0001_seed04.mp4,"An example English prompt copied exactly from the prompt suite."
B0001_seed05.mp4,"An example English prompt copied exactly from the prompt suite."
```

要求：

- CSV 必须保存为 UTF-8 或 UTF-8 with BOM；
- 表头必须是 `name,en_prompt`；
- `name` 必须包含扩展名，并与 `videos/` 中的文件名逐字一致；
- 每个 `name` 只能出现一次；
- 每个 prompt 必须恰好对应 5 个不同视频文件；
- prompt 中包含逗号时必须使用 CSV 双引号；
- `120-base_prompts.csv`、`200-aesthetic-prompts.csv`、`220-tag-prompts.csv` 使用官方文件的 `en_prompt` 列；
- `476-GQprompts.csv` 将官方 prompt suite 的 `prompt` 列原样复制到测试 CSV 的 `en_prompt` 列。

需要由参评者覆盖或重新生成的文件为：

```text
test/120-base_prompts/120-base_prompts.csv
test/200-aesthetic-prompts/200-aesthetic-prompts.csv
test/220-tag-prompts/220-tag-prompts.csv
test/476-GQprompts/476-GQprompts.csv
```

仓库内现有 `test/` 内容仅作为目录和格式参考，不应直接作为参评者最终提交数据。

### 6.4 支持的媒体格式

| 任务目录 | 支持格式 |
| --- | --- |
| `200-aesthetic-prompts` / VAQA | `.mp4`、`.webm`、`.gif`、`.avi`、`.mov` |
| `120-base_prompts` / VGQA | `.mp4`、`.webm`、`.gif`、`.webp`、`.avi`、`.mov` |
| `476-GQprompts` / VGQA | `.mp4`、`.webm`、`.gif`、`.webp`、`.avi`、`.mov` |
| `220-tag-prompts` / VTAG | `.mp4`、`.webm`、`.gif`、`.avi`、`.mov` |

扩展名必须与实际容器格式匹配。损坏、截断、无法完整解码或只有错误媒体流的文件会在推理前被拒绝。

## 7. `run_all.sh` 配置说明

### 7.1 运行标识与 GPU

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `GPU_ID` | `0` | 传给 `CUDA_VISIBLE_DEVICES`。单 GPU 示例为 `0`，多 GPU 可尝试 `0,1` |
| `MODEL_NAME` | `unnamed_model` | 待测视频生成模型名称，同时写入结果 CSV 和运行目录名 |
| `RUN_TIMESTAMP` | 自动生成 | 格式为 `YYYYMMDD-HHMMSS` |
| `RUN_DIR` | 自动生成 | `outputs/<模型名称>-<时间>/`，每次运行使用独立目录 |

模型名称中的 `/`、反斜杠和空格会被替换，以避免破坏输出目录结构。

### 7.2 VAQA 配置

| Shell 变量 | 默认配置 | 说明 |
| --- | --- | --- |
| `VAQA_VIDEO_ROOT_DIR` | `test/200-aesthetic-prompts` | VAQA 会递归扫描其中的 `videos/` 子目录 |
| `VAQA_OUTPUT_CSV` | `${RUN_DIR}/vaqa_pred_results.csv` | 每个成功视频的原始属性评分 |
| `VAQA_TEMP_FEAT_DIR` | `${RUN_DIR}/temp_features` | 临时视频特征目录 |
| `VAQA_DIM_OUTPUT_CSV` | `${RUN_DIR}/vaqa_dim_scores.csv` | VAQA 维度汇总分数 |
| `VAQA_VIDEO_EXTENSIONS` | `mp4 webm gif avi mov` | VAQA 接受的扩展名 |

VAQA 默认加载：

```text
VAQA/checkpoints/pytorch_model.bin.0
VAQA/checkpoints/通用属性分-model/best_model_epoch45_loss0.7096.pth
VAQA/checkpoints/人物属性分-model/best_model_epoch48_loss0.6425.pth
VAQA/modules/ViT-B-32.pt
```

### 7.3 VGQA 配置

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `VGQA_QO_CSV` | `VGQA/prompt_qo_deduped.csv` | prompt 与 q1–q6 问题的对应关系 |
| `VGQA_DATA_CSVS` | 120-base 与 476-GQ 两份测试 CSV | VGQA 输入 CSV，逗号分隔 |
| `VGQA_DATA_VIDEO_DIRS` | 120-base 与 476-GQ 两个视频目录 | 与 `VGQA_DATA_CSVS` 顺序一一对应 |
| `VGQA_OUTPUT_FILE` | `${RUN_DIR}/vgqa_pred_results.jsonl` | VGQA 原始模型预测 |
| `VGQA_DIM_OUTPUT_DIR` | `${RUN_DIR}` | VGQA 维度结果输出目录 |
| `VGQA_TEMP_DIR` | `${RUN_DIR}/vgqa_temp` | WebP 转 GIF 等临时文件目录 |
| `VGQA_BATCH_SIZE` | `2` | 当前配置中保留的批大小参数 |
| `QWEN_MODEL_PATH` | `Qwen3-VL-32B-Instruct` | Qwen 基础模型目录 |
| `VGQA_LORA_PATH` | `VGQA/v7-20260112-204659/checkpoint-570` | VGQA LoRA 目录 |
| `VGQA_MAX_TOKENS` | `512` | 单条预测最大生成 token 数 |
| `VGQA_TEMPERATURE` | `0` | VGQA 生成温度，默认确定性生成 |

VGQA 最终输出 31 个问题级子属性。

### 7.4 VTAG 配置

| 配置 | 默认值 | 说明 |
| --- | --- | --- |
| `VTAG_VIDEO_DIR` | `test/220-tag-prompts/videos` | VTAG 输入视频目录 |
| `VTAG_OUTPUT_PATH` | `${RUN_DIR}/vtag_pred_results.jsonl` | VTAG 原始属性预测 |
| `VTAG_DIM_OUTPUT_CSV` | `${RUN_DIR}/vtag_dim_scores.csv` | 各属性准确率 |
| `QWEN_MODEL_PATH` | `Qwen3-VL-32B-Instruct` | 与 VGQA 共用的基础模型目录 |
| `VTAG_LORA_PATH` | `VTAG/checkpoint-770` | VTAG LoRA 目录 |
| `VTAG_MAX_NEW_TOKENS` | `2048` | 单条预测最大新 token 数 |
| `VTAG_TEMPERATURE` | `0.001` | VTAG 生成温度 |
| `VTAG_TOP_P` | `1.0` | VTAG nucleus sampling 参数 |

VTAG 真值来自 `prompts/220-tag-prompts.csv`。

### 7.5 输入检查配置

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `VALIDATE_VIDEO_WORKERS` | `4` | 并行检查视频的 worker 数量 |
| `VALIDATE_VIDEO_TIMEOUT` | `300` | 每个文件 FFmpeg/FFprobe 检查超时秒数 |
| `FFMPEG_BIN` | 自动查找 | 自定义 FFmpeg 可执行文件路径 |
| `FFPROBE_BIN` | 自动查找 | 自定义 FFprobe 可执行文件路径 |

输入检查在任何模型加载之前执行，包括：

1. 测试 CSV 中的视频文件是否存在；
2. 文件扩展名与真实媒体格式是否匹配；
3. 视频是否能够完整解码、是否损坏；
4. 测试 CSV 是否覆盖对应 prompt suite 的全部 prompt。

检查失败时，本次运行立即停止，不执行 GPU 推理，也不发送邮件。详细问题写入 `${RUN_DIR}/validation_report.json`。

“每个 prompt 恰好 5 个视频”属于本评测的数据协议；当前输入检查主要验证文件有效性和 prompt 覆盖，参评者仍需自行确认每个 prompt 的视频数量。

### 7.6 邮件配置

邮件默认关闭。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SEND_EMAIL` | `false` | 只有设为 `true` 才发送邮件 |
| `SMTP_SERVER` | `smtp.126.com` | SMTP 服务器 |
| `SMTP_PORT` | `465` | 465 使用 SSL，其他端口使用 STARTTLS |
| `SENDER_EMAIL` | 空 | 发件邮箱 |
| `EMAIL_AUTH_CODE` | 空 | SMTP 授权码，不是邮箱登录密码 |
| `RECEIVER_EMAIL` | 脚本内默认值 | 收件邮箱；参评者应显式设置为自己的邮箱 |
| `EMAIL_SUBJECT` | `[CVPR评测结果] <MODEL_NAME> 评分报告` | 邮件主题 |
| `EMAIL_ATTACHMENTS` | 三个默认结果文件 | 逗号分隔的附件列表 |

启用邮件提交（`SEND_EMAIL=true`）即表示提交者同意公开以本次 `MODEL_NAME` 为模型名称的相关评测分数，并同意将这些分数加入 VGA-Bench/VGA-BenchV2 官方 **排行榜（Leaderboard）**。如果不同意公开分数，请勿启用邮件提交。提交者应确保模型名称准确，并确认自己有权提交和公开相关结果。

不要把邮箱密码或授权码写入代码、README、提交记录或公开日志。推荐只在当前终端临时设置环境变量。

## 8. 运行评测

### 8.1 基本运行

```bash
cd /path/to/codes_cvpr
MODEL_NAME=my_video_model GPU_ID=0 bash run_all.sh
```

### 8.2 自定义模型路径

```bash
MODEL_NAME=my_video_model \
GPU_ID=0 \
QWEN_MODEL_PATH=/data/models/Qwen3-VL-32B-Instruct \
VGQA_LORA_PATH=/data/models/v2-vgqa-lora \
VTAG_LORA_PATH=/data/models/v2-vtag-lora \
bash run_all.sh
```

### 8.3 启用邮件

```bash
MODEL_NAME=my_video_model \
GPU_ID=0 \
SEND_EMAIL=true \
SMTP_SERVER=smtp.126.com \
SMTP_PORT=465 \
SENDER_EMAIL=your_account@126.com \
EMAIL_AUTH_CODE='your_smtp_authorization_code' \
RECEIVER_EMAIL=your_receiver@example.com \
bash run_all.sh
```

## 9. 输出文件

每次运行都会创建独立目录：

```text
outputs/<MODEL_NAME>-<YYYYMMDD-HHMMSS>/
```

主要文件如下：

| 文件 | 内容 |
| --- | --- |
| `validation_report.json` | 输入视频、媒体格式、损坏情况和 prompt 覆盖检查 |
| `vaqa_pred_results.csv` | VAQA 每个视频的原始评分 |
| `vaqa_dim_scores.csv` | VAQA 各维度平均分 |
| `vgqa_pred_results.jsonl` | VGQA 每个视频的问答结果或错误信息 |
| `vgqa_dim_scores.csv` | 31 个 VGQA 问题级属性分数 |
| `vgqa_dim_scores.json` | VGQA 详细统计与错误计数 |
| `vtag_pred_results.jsonl` | VTAG 每个视频的属性预测 |
| `vtag_dim_scores.csv` | VTAG 各属性准确率和总体准确率 |
| `dimension_scores_summary.csv` | 三项任务的完整维度汇总 |
| `model_scores.csv` | VAQA、VGQA、VTAG 三项总分 |

## 10. 常见问题

### 找不到 Qwen 模型

确认以下目录存在：

```text
Qwen3-VL-32B-Instruct/
```

或显式设置：

```bash
export QWEN_MODEL_PATH=/absolute/path/to/Qwen3-VL-32B-Instruct
```

### CSV 中有视频，但脚本提示文件不存在

检查：

- `name` 是否包含正确扩展名；
- 大小写是否完全一致；
- 视频是否放在该 suite 的 `videos/` 目录；
- CSV 是否保存为 UTF-8；
- 文件名是否包含不可见空格。

### Prompt coverage 检查失败

必须从官方 prompt suite 原样复制英文 prompt。不要重新翻译、润色或改写。对于 476-GQ，将官方 `prompt` 列复制到测试 CSV 的 `en_prompt` 列。

### FFmpeg 检查超时

可以增加单文件超时时间：

```bash
VALIDATE_VIDEO_TIMEOUT=600 bash run_all.sh
```

### 显存不足

可尝试：

- 使用更大显存 GPU；
- 暴露多张 GPU，并确认 `device_map="auto"` 的分配结果；
- 减少同时运行的其他 GPU 任务；
- 检查是否误加载了多个基础模型副本。

## 11. 数据与安全说明

- 不要提交参评者的私有视频，除非已获得公开授权；
- 不要提交 SMTP 授权码、云盘令牌或其他凭证；
- 运行结果可能包含本地视频路径，公开结果前请检查并脱敏；
- 请确保待评测视频和使用的生成模型符合其许可证及数据使用条款。

## 12. Citation

如果本仓库对您的研究有所帮助，请引用：

**VGA-Bench: A Unified Benchmark and Multi-Model Framework for Video Aesthetics and Generation Quality Evaluation**  
*Longteng Jiang, DanDan Zheng, Qianqian Qiao, Heng Huang, Huaye Wang, Yihang Bo, Bao Peng, Jingdong Chen, Jun Zhou, Xin Jin*  
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition（CVPR）, 2026, pp. 30457–30466.

```bibtex
@inproceedings{jiang2026vgabench,
  title     = {VGA-Bench: A Unified Benchmark and Multi-Model Framework for Video Aesthetics and Generation Quality Evaluation},
  author    = {Jiang, Longteng and Zheng, DanDan and Qiao, Qianqian and Huang, Heng and Wang, Huaye and Bo, Yihang and Peng, Bao and Chen, Jingdong and Zhou, Jun and Jin, Xin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {30457--30466},
  year      = {2026}
}
```

## 13. License

请在公开发布前在项目根目录添加明确的 `LICENSE` 文件，并分别确认以下内容的许可证：

- 本仓库源代码；
- Qwen3-VL-32B-Instruct；
- VGQA/VTAG LoRA 权重；
- VAQA 与 CLIP 权重；
- prompt suite 和测试数据。

如果不同权重使用不同许可证，请在模型下载目录或模型卡中分别说明。
