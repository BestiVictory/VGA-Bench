#!/usr/bin/env python3
"""
发送邮件脚本，用于将结果 CSV 文件发送到指定邮箱。
支持 SMTP 授权码认证。
"""

import os
import sys
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime


def send_results_email(
    smtp_server,
    smtp_port,
    sender_email,
    sender_password,
    receiver_email,
    subject,
    body,
    attachments,
):
    """
    发送带附件的邮件。

    Args:
        smtp_server: SMTP 服务器地址
        smtp_port: SMTP 端口
        sender_email: 发件人邮箱
        sender_password: 授权码/密码
        receiver_email: 收件人邮箱
        subject: 邮件主题
        body: 邮件正文
        attachments: 附件文件路径列表
    """
    # 创建邮件对象
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    # 添加邮件正文
    msg.attach(MIMEText(body, "plain", "utf-8"))

    # 添加附件
    for file_path in attachments:
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                filename = os.path.basename(file_path)
                part.add_header(
                    "Content-Disposition",
                    f'attachment; filename="{filename}"',
                )
                msg.attach(part)
            print(f"[INFO] 已添加附件: {file_path}")
        else:
            print(f"[WARNING] 附件不存在，已跳过: {file_path}")

    # 连接 SMTP 服务器并发送邮件
    try:
        # 根据端口选择连接方式：465端口使用SSL，587端口使用STARTTLS
        print(f"[INFO] 正在连接 SMTP 服务器: {smtp_server}:{smtp_port}...")
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30)
        else:
            server = smtplib.SMTP(smtp_server, smtp_port, timeout=30)
            server.starttls()  # 启用 TLS 加密
        print(f"[INFO] 正在登录...")
        server.login(sender_email, sender_password)
        print(f"[INFO] 正在发送邮件...")
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        print(f"[SUCCESS] 邮件已成功发送到 {receiver_email}")
        return True
    except Exception as e:
        print(f"[ERROR] 邮件发送失败: {e}")
        return False


def main():
    # 从环境变量读取配置
    smtp_server = os.getenv("SMTP_SERVER", "smtp.qq.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    sender_email = os.getenv("SENDER_EMAIL", "")
    sender_password = os.getenv("EMAIL_AUTH_CODE", "")  # 授权码
    receiver_email = os.getenv("RECEIVER_EMAIL", "")

    # 检查必要配置
    if not sender_email:
        print("[ERROR] 未设置发件人邮箱 (SENDER_EMAIL)")
        sys.exit(1)
    if not sender_password:
        print("[ERROR] 未设置邮箱授权码 (EMAIL_AUTH_CODE)")
        sys.exit(1)
    if not receiver_email:
        print("[ERROR] 未设置收件人邮箱 (RECEIVER_EMAIL)")
        sys.exit(1)

    # 获取要发送的附件列表
    script_dir = os.path.dirname(os.path.abspath(__file__))
    attachments_str = os.getenv(
        "EMAIL_ATTACHMENTS",
        f"{script_dir}/model_scores.csv,{script_dir}/dimension_scores_summary.csv",
    )
    attachments = [f.strip() for f in attachments_str.split(",") if f.strip()]

    # 构建邮件主题和正文
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = os.getenv("MODEL_NAME", "unnamed_model")
    subject = os.getenv("EMAIL_SUBJECT", f"[CVPR评测结果] {model_name} 评分报告 - {timestamp}")

    default_body = f"""您好，

CVPR 评测任务已完成，请查收附件中的结果文件。

模型名称: {model_name}
生成时间: {timestamp}
附件说明:
- model_scores.csv: 模型总分汇总
- dimension_scores_summary.csv: 各维度分数汇总

如有问题，请联系管理员。
"""
    body = os.getenv("EMAIL_BODY", default_body)

    # 发送邮件
    success = send_results_email(
        smtp_server=smtp_server,
        smtp_port=smtp_port,
        sender_email=sender_email,
        sender_password=sender_password,
        receiver_email=receiver_email,
        subject=subject,
        body=body,
        attachments=attachments,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
