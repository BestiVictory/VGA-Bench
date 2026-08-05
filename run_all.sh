#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ================================================================
#                        GPU 配置（单卡推理）
# ================================================================
GPU_ID="${GPU_ID:-0}"

# ================================================================
#                        本次运行目录
# ================================================================
MODEL_NAME="${MODEL_NAME:-unnamed_model}"
SAFE_MODEL_NAME="${MODEL_NAME//\//_}"
SAFE_MODEL_NAME="${SAFE_MODEL_NAME//\\/_}"
SAFE_MODEL_NAME="${SAFE_MODEL_NAME// /_}"
RUN_TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
RUN_DIR="${SCRIPT_DIR}/outputs/${SAFE_MODEL_NAME}-${RUN_TIMESTAMP}"
mkdir -p "${RUN_DIR}"

echo "本次运行目录: ${RUN_DIR}"

# ================================================================
#                        VAQA 配置区
# ================================================================
VAQA_VIDEO_ROOT_DIR="${SCRIPT_DIR}/test/200-aesthetic-prompts"
VAQA_OUTPUT_CSV="${RUN_DIR}/vaqa_pred_results.csv"
VAQA_TEMP_FEAT_DIR="${RUN_DIR}/temp_features"
VAQA_DIM_OUTPUT_CSV="${RUN_DIR}/vaqa_dim_scores.csv"
VAQA_VIDEO_EXTENSIONS="mp4 webm gif avi mov"

# ================================================================
#                        VGQA 配置区
# ================================================================
VGQA_QO_CSV="${SCRIPT_DIR}/VGQA/prompt_qo_deduped.csv"
VGQA_DATA_CSVS="${SCRIPT_DIR}/test/120-base_prompts/120-base_prompts.csv,${SCRIPT_DIR}/test/476-GQprompts/476-GQprompts.csv"
VGQA_DATA_VIDEO_DIRS="${SCRIPT_DIR}/test/120-base_prompts/videos,${SCRIPT_DIR}/test/476-GQprompts/videos"
VGQA_OUTPUT_FILE="${RUN_DIR}/vgqa_pred_results.jsonl"
VGQA_DIM_OUTPUT_DIR="${RUN_DIR}"
VGQA_TEMP_DIR="${RUN_DIR}/vgqa_temp"
VGQA_BATCH_SIZE=2

# ================================================================
#                        VTAG 配置区
# ================================================================
VTAG_VIDEO_DIR="${SCRIPT_DIR}/test/220-tag-prompts/videos"
VTAG_OUTPUT_PATH="${RUN_DIR}/vtag_pred_results.jsonl"
VTAG_DIM_OUTPUT_CSV="${RUN_DIR}/vtag_dim_scores.csv"

# ================================================================
#                      邮件发送配置区
# ================================================================
# 是否发送邮件: true/false（默认false，需显式设置为true才发送）
SEND_EMAIL="${SEND_EMAIL:-false}"

# SMTP 服务器配置（默认使用 QQ 邮箱）
SMTP_SERVER="${SMTP_SERVER:-smtp.126.com}"
SMTP_PORT="${SMTP_PORT:-465}"

# 发件人邮箱和授权码（请通过环境变量设置，避免硬编码）
# 授权码获取方式（以 QQ 邮箱为例）：
# 1. 登录 QQ 邮箱网页版
# 2. 设置 -> 账户 -> 开启 POP3/SMTP 服务
# 3. 获取 16 位授权码（不是邮箱登录密码）
SENDER_EMAIL="${SENDER_EMAIL:-}"
EMAIL_AUTH_CODE="${EMAIL_AUTH_CODE:-}"  # 16位授权码，不是邮箱密码

# 收件人邮箱（多个用逗号分隔）
RECEIVER_EMAIL="${RECEIVER_EMAIL:-jang_lt@qq.com}"

# 邮件主题和附件
EMAIL_SUBJECT="${EMAIL_SUBJECT:-[CVPR评测结果] ${MODEL_NAME} 评分报告}"
EMAIL_ATTACHMENTS="${EMAIL_ATTACHMENTS:-${RUN_DIR}/model_scores.csv,${RUN_DIR}/dimension_scores_summary.csv,${RUN_DIR}/validation_report.json}"

# ================================================================
#                  推理前检查测试输入完整性
# ================================================================
echo ""
echo "========== 测试视频与 prompt suite 检查开始 =========="
cd "${SCRIPT_DIR}"
if ! python validate_run.py \
    --run-dir "${RUN_DIR}" \
    --report "${RUN_DIR}/validation_report.json"; then
    echo "[ERROR] 测试视频或 prompt suite 检查未通过，已终止本次运行。"
    echo "        请查看: ${RUN_DIR}/validation_report.json"
    exit 1
fi
echo "========== 测试视频与 prompt suite 检查完成 =========="
echo "[PASS] 测试视频存在性、格式、损坏情况和 prompt suite 覆盖检查通过。"

# ================================================================
#                      执行 VAQA 推理
# ================================================================
echo "========== VAQA 推理开始 =========="
cd "${SCRIPT_DIR}/VAQA"
CUDA_VISIBLE_DEVICES="${GPU_ID}" python 1predict_scores-TV.py \
    --video_root_dir "${VAQA_VIDEO_ROOT_DIR}" \
    --output_csv "${VAQA_OUTPUT_CSV}" \
    --temp_feat_dir "${VAQA_TEMP_FEAT_DIR}" \
    --video_extensions ${VAQA_VIDEO_EXTENSIONS}
echo "========== VAQA 推理完成 =========="

# ================================================================
#                      执行 VGQA 推理
# ================================================================
echo "========== VGQA 推理开始 =========="
cd "${SCRIPT_DIR}/VGQA"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
VGQA_OUTPUT_FILE="${VGQA_OUTPUT_FILE}" \
VGQA_TEMP_DIR="${VGQA_TEMP_DIR}" \
VGQA_BATCH_SIZE="${VGQA_BATCH_SIZE}" \
VGQA_QO_CSV="${VGQA_QO_CSV}" \
VGQA_DATA_CSVS="${VGQA_DATA_CSVS}" \
VGQA_DATA_VIDEO_DIRS="${VGQA_DATA_VIDEO_DIRS}" \
MODEL_NAME="${MODEL_NAME}" \
python run_all_pred.py
echo "========== VGQA 推理完成 =========="

# ================================================================
#                      执行 VTAG 推理
# ================================================================
echo "========== VTAG 推理开始 =========="
cd "${SCRIPT_DIR}/VTAG"
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
VTAG_VIDEO_DIR="${VTAG_VIDEO_DIR}" \
VTAG_OUTPUT_PATH="${VTAG_OUTPUT_PATH}" \
python run_aesthetic_pred.py
echo "========== VTAG 推理完成 =========="

# ================================================================
#                      执行各任务维度分数计算
# ================================================================
echo ""
echo "========== VAQA 维度分数计算开始 =========="
cd "${SCRIPT_DIR}/VAQA"
VAQA_PRED_PATH="${VAQA_OUTPUT_CSV}" \
VAQA_DIM_OUTPUT_PATH="${VAQA_DIM_OUTPUT_CSV}" \
python calculate_dimension_scores.py
echo "========== VAQA 维度分数计算完成 =========="

echo ""
echo "========== VGQA 维度分数分析开始 =========="
cd "${SCRIPT_DIR}/VGQA"
VGQA_INPUT_FILE="${VGQA_OUTPUT_FILE}" \
VGQA_SCORE_OUTPUT_DIR="${VGQA_DIM_OUTPUT_DIR}" \
python vgqa_analyze.py
echo "========== VGQA 维度分数分析完成 =========="

echo ""
echo "========== VTAG 准确率计算开始 =========="
cd "${SCRIPT_DIR}/VTAG"
VTAG_PRED_FILE="${VTAG_OUTPUT_PATH}" \
VTAG_SCORE_OUTPUT_PATH="${VTAG_DIM_OUTPUT_CSV}" \
python calc_accuracy.py
echo "========== VTAG 准确率计算完成 =========="

# ================================================================
#                      汇总各任务分数
# ================================================================
echo ""
echo "========== 汇总分数开始 =========="
cd "${SCRIPT_DIR}"
MODEL_NAME="${MODEL_NAME}" \
VAQA_SCORE_CSV="${VAQA_DIM_OUTPUT_CSV}" \
VGQA_SCORE_CSV="${RUN_DIR}/vgqa_dim_scores.csv" \
VTAG_SCORE_CSV="${VTAG_DIM_OUTPUT_CSV}" \
SUMMARY_OUTPUT_DIR="${RUN_DIR}" \
python summarize_scores.py
echo "========== 汇总分数完成 =========="

echo ""
echo "============================================================"
echo "推理、评分、校验与汇总流程执行完毕"
echo "============================================================"
echo "VAQA 输出: ${VAQA_OUTPUT_CSV}"
echo "VGQA 输出: ${VGQA_OUTPUT_FILE}"
echo "VTAG 输出: ${VTAG_OUTPUT_PATH}"
echo "VAQA 维度分数: ${VAQA_DIM_OUTPUT_CSV}"
echo "VGQA 维度分数: ${RUN_DIR}/vgqa_dim_scores.csv"
echo "VTAG 维度分数: ${VTAG_DIM_OUTPUT_CSV}"
echo "汇总分数: ${RUN_DIR}/dimension_scores_summary.csv"
echo "简洁总分: ${RUN_DIR}/model_scores.csv"
echo "完整性报告: ${RUN_DIR}/validation_report.json"

# ================================================================
#                      发送结果邮件
# ================================================================
echo ""
echo "========== 邮件发送 =========="

# 检查是否启用邮件发送
if [ "${SEND_EMAIL}" != "true" ]; then
    echo "[INFO] SEND_EMAIL=${SEND_EMAIL}，跳过邮件发送"
    echo "      如需发送邮件，请设置: SEND_EMAIL=true"
    exit 0
fi

# 检查是否配置了邮箱
if [ -z "${SENDER_EMAIL}" ] || [ -z "${EMAIL_AUTH_CODE}" ] || [ -z "${RECEIVER_EMAIL}" ]; then
    echo "[WARN] 未配置邮箱信息，跳过邮件发送"
    echo "      请设置以下环境变量以启用邮件功能:"
    echo "      - SENDER_EMAIL: 发件人邮箱"
    echo "      - EMAIL_AUTH_CODE: 邮箱授权码"
    echo "      - RECEIVER_EMAIL: 收件人邮箱"
    exit 0
fi

# 执行邮件发送
cd "${SCRIPT_DIR}"
SMTP_SERVER="${SMTP_SERVER}" \
SMTP_PORT="${SMTP_PORT}" \
SENDER_EMAIL="${SENDER_EMAIL}" \
EMAIL_AUTH_CODE="${EMAIL_AUTH_CODE}" \
RECEIVER_EMAIL="${RECEIVER_EMAIL}" \
EMAIL_SUBJECT="${EMAIL_SUBJECT}" \
EMAIL_ATTACHMENTS="${EMAIL_ATTACHMENTS}" \
MODEL_NAME="${MODEL_NAME}" \
python send_email.py

if [ $? -eq 0 ]; then
    echo "========== 邮件发送成功 =========="
else
    echo "========== 邮件发送失败 =========="
    exit 1
fi
