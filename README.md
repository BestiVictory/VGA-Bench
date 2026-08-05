# VGA-Bench and VGA-BenchV2: Video Aesthetics and Generation Quality Evaluation

**English** | [中文](README_CN.md)

This repository provides video-generation evaluation components associated with two related works:

- **VGA-Bench: A Unified Benchmark and Multi-Model Framework for Video Aesthetics and Generation Quality Evaluation**;
- **VGA-BenchV2: An Expanded Unified Benchmark and Multi-Model Framework for Evaluating Video Aesthetics and Generation Quality** (currently unpublished).

Participants generate videos with their own models using the four official prompt suites, prepare the required videos and CSV mappings under `test/`, and run a single shell script to perform input validation, three evaluation tasks, score aggregation, and optional email notification.

> This repository contains the evaluation pipeline only. It does not include a candidate text-to-video generation model. Participants must generate their own videos and prepare the contents of the `test/` directory.

## 1. About VGA-Bench and VGA-BenchV2

VGA-Bench evaluates text-to-video generation models from complementary perspectives instead of relying on a single overall metric. VGA-BenchV2 expands this unified benchmark and multi-model evaluation framework for video aesthetics and generation quality. This repository contains evaluation components used by both works. The released pipeline contains three tracks:

| Track | Prompt suite | Evaluation focus | Output |
| --- | --- | --- | --- |
| VAQA | `200-aesthetic-prompts` | Overall aesthetics, composition, shot size, lighting, visual tone, color, depth of field, expression, costume, and makeup | Normalized dimension scores and a VAQA overall score |
| VGQA | `120-base_prompts`, `476-GQprompts` | Question-level assessment of basic visual quality, temporal stability, motion, scene quality, and general generation quality | 31 question-level attribute scores and a VGQA overall score |
| VTAG | `220-tag-prompts` | Alignment between the prompt and controllable visual attributes such as composition, lighting, shot size, depth of field, saturation, brightness, color temperature, and contrast | Accuracy for 11 visual attributes and a VTAG overall score |

The shared idea of VGA-Bench and VGA-BenchV2 is to model aesthetic quality, general video generation quality, and controllable visual-attribute alignment separately, then report interpretable dimension-level results. This repository corresponds to their evaluation components and does not include the papers' training or video-generation code.

VGA-Bench was published in the **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026**. VGA-BenchV2 has not yet been published. The [Citation](#13-citation) section therefore remains the citation for the published VGA-Bench paper.

## 2. Repository Structure

```text
.
├── prompts/                         # Official prompt suites and ground truth
│   ├── 120-base_prompts.csv
│   ├── 200-aesthetic-prompts.csv
│   ├── 220-tag-prompts.csv
│   └── 476-GQprompts.csv
├── test/                            # Participant-provided videos and CSV mappings
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
├── VAQA/                            # Aesthetic and character-attribute scoring
├── VGQA/                            # General video-quality question answering
├── VTAG/                            # Visual-tag alignment evaluation
├── validate_run.py                 # Pre-inference input validation
├── summarize_scores.py             # Aggregates the three task scores
├── send_email.py                   # Optional result email sender
├── run_all.sh                      # Main evaluation entry point
├── requirements.txt
├── environment.yml
├── README.md                       # English documentation
└── README_CN.md                    # Chinese documentation
```

## 3. System and Hardware Requirements

Recommended environment:

- Linux or WSL2;
- Python 3.10;
- an NVIDIA driver compatible with CUDA 12.4;
- FFmpeg and FFprobe;
- sufficient disk space for Qwen3-VL-32B, the evaluation weights, 5,080 generated videos, temporary files, and evaluation outputs.

VGQA and VTAG use Qwen3-VL-32B-Instruct. A 32B BF16 model has substantial VRAM requirements, which vary with the model build, video length, sampled frame count, and inference framework. A high-memory GPU is recommended. To expose multiple GPUs, `GPU_ID` may be set to a comma-separated list such as `0,1`, but users should verify that the current inference framework distributes the model correctly.

## 4. Environment Setup

### 4.1 Conda Installation (Recommended)

Run the following commands from the repository root:

```bash
conda env create -f environment.yml
conda activate cvpr-video-eval
```

`environment.yml` installs Python 3.10 and FFmpeg, then uses the root-level `requirements.txt` to install the Python dependencies.

### 4.2 pip Installation

If compatible versions of Python 3.10, CUDA, FFmpeg, and FFprobe are already installed:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify FFmpeg and FFprobe:

```bash
ffmpeg -version
ffprobe -version
```

## 5. Download the Base Model and Evaluation Weights

All required evaluation models and weights are available from Google Drive:

**[Download the VGA-Bench and VGA-BenchV2 models and evaluation weights](https://drive.google.com/drive/folders/18j3ExC10LgmnE47k70YvATtXNbhnogMW?usp=drive_link)**

After downloading, place the files in the following structure. Do not download an `adapter_model.safetensors` file without its corresponding `adapter_config.json`.

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

If the files are stored elsewhere, specify their locations with environment variables:

```bash
export QWEN_MODEL_PATH=/absolute/path/to/Qwen3-VL-32B-Instruct
export VGQA_LORA_PATH=/absolute/path/to/VGQA/checkpoint-570
export VTAG_LORA_PATH=/absolute/path/to/VTAG/checkpoint-770
```

## 6. Prepare Evaluation Videos

### 6.1 Generate Five Videos for Every Prompt

Participants must generate **five videos for every prompt** in each prompt suite. The five videos may use different random seeds, but they should be generated by the same candidate model using a consistent evaluation configuration.

| Prompt suite | Number of prompts | Videos per prompt | Required videos |
| --- | ---: | ---: | ---: |
| `120-base_prompts` | 120 | 5 | 600 |
| `200-aesthetic-prompts` | 200 | 5 | 1,000 |
| `220-tag-prompts` | 220 | 5 | 1,100 |
| `476-GQprompts` | 476 | 5 | 2,380 |
| **Total** | **1,016** | **5** | **5,080** |

The evaluation mapping uses the English prompt text as its key. Copy prompts exactly from the official prompt-suite files. Do not rewrite, translate, re-capitalize, or modify punctuation.

### 6.2 Video Directories

Place the generated videos in the corresponding directory:

```text
test/120-base_prompts/videos/
test/200-aesthetic-prompts/videos/
test/220-tag-prompts/videos/
test/476-GQprompts/videos/
```

Every filename must be unique within its prompt-suite directory. A recommended naming pattern is:

```text
B0001_seed01.mp4
B0001_seed02.mp4
B0001_seed03.mp4
B0001_seed04.mp4
B0001_seed05.mp4
```

Filenames do not have to match the prompt IDs, but they must match the `name` values in the corresponding test CSV exactly.

### 6.3 Test CSV Format

Every generated video must occupy one row in its corresponding test CSV. All four test CSV files use the same two-column format:

```csv
name,en_prompt
B0001_seed01.mp4,"An example English prompt copied exactly from the prompt suite."
B0001_seed02.mp4,"An example English prompt copied exactly from the prompt suite."
B0001_seed03.mp4,"An example English prompt copied exactly from the prompt suite."
B0001_seed04.mp4,"An example English prompt copied exactly from the prompt suite."
B0001_seed05.mp4,"An example English prompt copied exactly from the prompt suite."
```

Requirements:

- save each CSV as UTF-8 or UTF-8 with BOM;
- the header must be exactly `name,en_prompt`;
- `name` must include the file extension and exactly match a file in `videos/`;
- each `name` may appear only once;
- every prompt must correspond to exactly five different video files;
- prompts containing commas must be enclosed in CSV double quotes;
- for `120-base_prompts`, `200-aesthetic-prompts`, and `220-tag-prompts`, copy the official `en_prompt` column;
- for `476-GQprompts`, copy the official `prompt` column into the test CSV's `en_prompt` column.

Participants must replace or regenerate these files:

```text
test/120-base_prompts/120-base_prompts.csv
test/200-aesthetic-prompts/200-aesthetic-prompts.csv
test/220-tag-prompts/220-tag-prompts.csv
test/476-GQprompts/476-GQprompts.csv
```

The existing contents under `test/` are directory and format references only. They should not be treated as a participant's final evaluation submission.

### 6.4 Supported Media Formats

| Evaluation directory | Supported formats |
| --- | --- |
| `200-aesthetic-prompts` / VAQA | `.mp4`, `.webm`, `.gif`, `.avi`, `.mov` |
| `120-base_prompts` / VGQA | `.mp4`, `.webm`, `.gif`, `.webp`, `.avi`, `.mov` |
| `476-GQprompts` / VGQA | `.mp4`, `.webm`, `.gif`, `.webp`, `.avi`, `.mov` |
| `220-tag-prompts` / VTAG | `.mp4`, `.webm`, `.gif`, `.avi`, `.mov` |

The file extension must match the actual container format. Corrupted, truncated, partially decodable, or incorrectly labeled media files are rejected before inference.

## 7. `run_all.sh` Configuration

### 7.1 Run Identity and GPU Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `GPU_ID` | `0` | Passed to `CUDA_VISIBLE_DEVICES`. Use `0` for one GPU or, where supported, `0,1` for multiple GPUs |
| `MODEL_NAME` | `unnamed_model` | Name of the candidate video-generation model; recorded in output CSVs and the run-directory name |
| `RUN_TIMESTAMP` | generated automatically | Uses the format `YYYYMMDD-HHMMSS` |
| `RUN_DIR` | generated automatically | `outputs/<model-name>-<timestamp>/`; every run uses a new directory |

Forward slashes, backslashes, and spaces in `MODEL_NAME` are replaced when the output-directory name is constructed.

### 7.2 VAQA Configuration

| Shell variable | Default | Description |
| --- | --- | --- |
| `VAQA_VIDEO_ROOT_DIR` | `test/200-aesthetic-prompts` | VAQA recursively scans the `videos/` subdirectory |
| `VAQA_OUTPUT_CSV` | `${RUN_DIR}/vaqa_pred_results.csv` | Raw per-video VAQA predictions |
| `VAQA_TEMP_FEAT_DIR` | `${RUN_DIR}/temp_features` | Temporary video-feature directory |
| `VAQA_DIM_OUTPUT_CSV` | `${RUN_DIR}/vaqa_dim_scores.csv` | Aggregated VAQA dimension scores |
| `VAQA_VIDEO_EXTENSIONS` | `mp4 webm gif avi mov` | Extensions accepted by VAQA |

VAQA loads the following files by default:

```text
VAQA/checkpoints/pytorch_model.bin.0
VAQA/checkpoints/通用属性分-model/best_model_epoch45_loss0.7096.pth
VAQA/checkpoints/人物属性分-model/best_model_epoch48_loss0.6425.pth
VAQA/modules/ViT-B-32.pt
```

### 7.3 VGQA Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `VGQA_QO_CSV` | `VGQA/prompt_qo_deduped.csv` | Mapping from each prompt to its q1–q6 questions |
| `VGQA_DATA_CSVS` | the 120-base and 476-GQ test CSV files | Comma-separated VGQA input CSVs |
| `VGQA_DATA_VIDEO_DIRS` | the 120-base and 476-GQ video directories | Must correspond to `VGQA_DATA_CSVS` in the same order |
| `VGQA_OUTPUT_FILE` | `${RUN_DIR}/vgqa_pred_results.jsonl` | Raw VGQA model predictions |
| `VGQA_DIM_OUTPUT_DIR` | `${RUN_DIR}` | VGQA score-output directory |
| `VGQA_TEMP_DIR` | `${RUN_DIR}/vgqa_temp` | Temporary files, including WebP-to-GIF conversions |
| `VGQA_BATCH_SIZE` | `2` | Batch-size setting retained by the current pipeline |
| `QWEN_MODEL_PATH` | `Qwen3-VL-32B-Instruct` | Qwen base-model directory |
| `VGQA_LORA_PATH` | `VGQA/v7-20260112-204659/checkpoint-570` | VGQA LoRA directory |
| `VGQA_MAX_TOKENS` | `512` | Maximum generated tokens per prediction |
| `VGQA_TEMPERATURE` | `0` | Generation temperature; the default is deterministic |

VGQA reports 31 question-level attributes from `VGQA/index.json`, excluding index 23, “Is the video content aesthetically pleasing?”. The normalized scores of the available attributes are averaged to produce the VGQA overall score.

### 7.4 VTAG Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `VTAG_VIDEO_DIR` | `test/220-tag-prompts/videos` | VTAG input-video directory |
| `VTAG_OUTPUT_PATH` | `${RUN_DIR}/vtag_pred_results.jsonl` | Raw VTAG attribute predictions |
| `VTAG_DIM_OUTPUT_CSV` | `${RUN_DIR}/vtag_dim_scores.csv` | Per-attribute accuracy output |
| `QWEN_MODEL_PATH` | `Qwen3-VL-32B-Instruct` | Base model shared with VGQA |
| `VTAG_LORA_PATH` | `VTAG/checkpoint-770` | VTAG LoRA directory |
| `VTAG_MAX_NEW_TOKENS` | `2048` | Maximum number of newly generated tokens per prediction |
| `VTAG_TEMPERATURE` | `0.001` | VTAG generation temperature |
| `VTAG_TOP_P` | `1.0` | VTAG nucleus-sampling parameter |

VTAG ground truth is read from `prompts/220-tag-prompts.csv`.

### 7.5 Input Validation Configuration

| Environment variable | Default | Description |
| --- | --- | --- |
| `VALIDATE_VIDEO_WORKERS` | `4` | Number of parallel media-validation workers |
| `VALIDATE_VIDEO_TIMEOUT` | `300` | FFmpeg/FFprobe timeout in seconds for each file |
| `FFMPEG_BIN` | automatically detected | Optional custom FFmpeg executable path |
| `FFPROBE_BIN` | automatically detected | Optional custom FFprobe executable path |

Input validation is performed before any model is loaded. It checks:

1. whether every video listed in the test CSV exists;
2. whether the extension matches the actual media/container format;
3. whether the complete video can be decoded without corruption;
4. whether the test CSV covers every prompt in the corresponding prompt suite.

If validation fails, the run stops before GPU inference and no email is sent. Detailed issues are written to `${RUN_DIR}/validation_report.json`.

The requirement of exactly five videos per prompt is part of the evaluation protocol. The current input validator primarily checks media validity and prompt coverage, so participants should also verify the per-prompt video count when preparing their submission.

### 7.6 Email Configuration

Email delivery is disabled by default.

| Environment variable | Default | Description |
| --- | --- | --- |
| `SEND_EMAIL` | `false` | Email is sent only when set to `true` |
| `SMTP_SERVER` | `smtp.126.com` | SMTP server |
| `SMTP_PORT` | `465` | Port 465 uses SSL; other ports use STARTTLS |
| `SENDER_EMAIL` | empty | Sender email address |
| `EMAIL_AUTH_CODE` | empty | SMTP authorization code, not the mailbox login password |
| `RECEIVER_EMAIL` | the value configured in the script | Recipient address; participants should explicitly set their own address |
| `EMAIL_SUBJECT` | `[CVPR Evaluation Results] <MODEL_NAME> Score Report` | Email subject concept; the current script uses a Chinese default subject |
| `EMAIL_ATTACHMENTS` | three default result files | Comma-separated attachment paths |

Never commit a mailbox password, SMTP authorization code, cloud-drive token, or other credential to the repository, documentation, or public logs. Set credentials only in the current shell environment.

## 8. Run the Evaluation

### 8.1 Basic Run

```bash
cd /path/to/codes_cvpr
MODEL_NAME=my_video_model GPU_ID=0 bash run_all.sh
```

### 8.2 Custom Model Locations

```bash
MODEL_NAME=my_video_model \
GPU_ID=0 \
QWEN_MODEL_PATH=/data/models/Qwen3-VL-32B-Instruct \
VGQA_LORA_PATH=/data/models/vga-bench-vgqa-lora \
VTAG_LORA_PATH=/data/models/vga-bench-vtag-lora \
bash run_all.sh
```

### 8.3 Enable Email Delivery

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

## 9. Output Files

Every run creates an isolated output directory:

```text
outputs/<MODEL_NAME>-<YYYYMMDD-HHMMSS>/
```

| File | Description |
| --- | --- |
| `validation_report.json` | Input existence, media-format, corruption, and prompt-coverage report |
| `vaqa_pred_results.csv` | Raw per-video VAQA scores |
| `vaqa_dim_scores.csv` | Aggregated VAQA dimension scores |
| `vgqa_pred_results.jsonl` | Per-video VGQA answers or error records |
| `vgqa_dim_scores.csv` | Scores for the 31 VGQA question-level attributes |
| `vgqa_dim_scores.json` | Detailed VGQA statistics and error counts |
| `vtag_pred_results.jsonl` | Per-video VTAG attribute predictions |
| `vtag_dim_scores.csv` | VTAG per-attribute and overall accuracy |
| `dimension_scores_summary.csv` | Full dimension-level summary for all three tasks |
| `model_scores.csv` | VAQA, VGQA, and VTAG overall scores |

## 10. Score Interpretation

### VAQA

VAQA produces attribute scores on a 0–10 scale. Dimension aggregation divides these values by 10 to normalize them to 0–1. The VAQA overall score uses the `overall` dimension.

### VGQA

VGQA maps the q1–q6 answers associated with each prompt back to `VGQA/index.json`. It retains 31 question-level attributes, normalizes their scores, and computes an equal-weight average.

### VTAG

VTAG compares predicted visual attributes against `prompts/220-tag-prompts.csv`. For multi-label attributes, every ground-truth label must be included in the prediction set. The `overall` row in `vtag_dim_scores.csv` is not counted again when the VTAG task score is summarized.

The three task scores are reported separately. Do not combine them into a new single ranking score unless the paper or an official evaluation protocol explicitly defines that aggregation.

## 11. Troubleshooting

### Qwen Model Cannot Be Found

Confirm that this directory exists:

```text
Qwen3-VL-32B-Instruct/
```

Alternatively:

```bash
export QWEN_MODEL_PATH=/absolute/path/to/Qwen3-VL-32B-Instruct
```

### A Video Is Listed in the CSV but Reported Missing

Check that:

- `name` includes the correct extension;
- filename capitalization matches exactly;
- the video is in the correct prompt suite's `videos/` directory;
- the CSV is encoded as UTF-8;
- the filename contains no leading, trailing, or invisible whitespace.

### Prompt Coverage Validation Fails

Copy the English prompt exactly from the official prompt suite. Do not translate or rewrite it. For 476-GQ, copy the official `prompt` column into the test CSV's `en_prompt` column.

### FFmpeg Validation Times Out

Increase the per-file timeout:

```bash
VALIDATE_VIDEO_TIMEOUT=600 bash run_all.sh
```

### Out of GPU Memory

Possible mitigations include:

- use a GPU with more memory;
- expose multiple GPUs and verify the `device_map="auto"` allocation;
- stop unrelated GPU workloads;
- confirm that the base model is not being loaded multiple times in the same process.

### Can `__pycache__` Be Deleted?

Yes. `__pycache__/` directories and `.pyc` files do not need to be distributed. Python recreates them automatically. Add the following entries to `.gitignore`:

```gitignore
__pycache__/
*.py[cod]
outputs/
```

## 12. Data and Security Notes

- Do not publish private participant videos without authorization.
- Do not commit SMTP authorization codes, cloud-drive tokens, or other secrets.
- Evaluation outputs may contain local video paths; inspect and sanitize them before publication.
- Ensure that the candidate videos, generation model, base models, adapters, and evaluation weights are used in accordance with their licenses and data-use terms.

## 13. Citation

If this repository is useful for your research, please cite:

**VGA-Bench: A Unified Benchmark and Multi-Model Framework for Video Aesthetics and Generation Quality Evaluation**  
*Longteng Jiang, DanDan Zheng, Qianqian Qiao, Heng Huang, Huaye Wang, Yihang Bo, Bao Peng, Jingdong Chen, Jun Zhou, Xin Jin*  
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 30457–30466.

```bibtex
@inproceedings{jiang2026vgabench,
  title     = {VGA-Bench: A Unified Benchmark and Multi-Model Framework for Video Aesthetics and Generation Quality Evaluation},
  author    = {Jiang, Longteng and Zheng, DanDan and Qiao, Qianqian and Huang, Heng and Wang, Huaye and Bo, Yihang and Peng, Bao and Chen, Jingdong and Zhou, Jun and Jin, Xin},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {30457--30466},
  year      = {2026}
}
```

## 14. License

Add an explicit `LICENSE` file before public release and verify the licenses that apply to:

- the source code in this repository;
- Qwen3-VL-32B-Instruct;
- the VGQA and VTAG LoRA adapters;
- the VAQA and CLIP weights;
- the prompt suites and any released evaluation data.

If different model files use different licenses, document them separately in the download directory or in accompanying model cards.
