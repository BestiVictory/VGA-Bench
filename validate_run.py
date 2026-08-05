#!/usr/bin/env python3
"""检查测试视频文件及 prompt suite 覆盖关系。"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_TERMINAL_ISSUES = 100

DATASETS = [
    {
        "name": "VAQA/200-aesthetic-prompts",
        "test_csv": PROJECT_ROOT / "test/200-aesthetic-prompts/200-aesthetic-prompts.csv",
        "video_dir": PROJECT_ROOT / "test/200-aesthetic-prompts/videos",
        "suite_csv": PROJECT_ROOT / "prompts/200-aesthetic-prompts.csv",
        "test_prompt_column": "en_prompt",
        "suite_prompt_column": "en_prompt",
        "extensions": {".mp4", ".webm", ".gif", ".avi", ".mov"},
    },
    {
        "name": "VGQA/120-base_prompts",
        "test_csv": PROJECT_ROOT / "test/120-base_prompts/120-base_prompts.csv",
        "video_dir": PROJECT_ROOT / "test/120-base_prompts/videos",
        "suite_csv": PROJECT_ROOT / "prompts/120-base_prompts.csv",
        "test_prompt_column": "en_prompt",
        "suite_prompt_column": "en_prompt",
        "extensions": {".mp4", ".webm", ".gif", ".webp", ".avi", ".mov"},
    },
    {
        "name": "VGQA/476-GQprompts",
        "test_csv": PROJECT_ROOT / "test/476-GQprompts/476-GQprompts.csv",
        "video_dir": PROJECT_ROOT / "test/476-GQprompts/videos",
        "suite_csv": PROJECT_ROOT / "prompts/476-GQprompts.csv",
        "test_prompt_column": "en_prompt",
        "suite_prompt_column": "prompt",
        "extensions": {".mp4", ".webm", ".gif", ".webp", ".avi", ".mov"},
    },
    {
        "name": "VTAG/220-tag-prompts",
        "test_csv": PROJECT_ROOT / "test/220-tag-prompts/220-tag-prompts.csv",
        "video_dir": PROJECT_ROOT / "test/220-tag-prompts/videos",
        "suite_csv": PROJECT_ROOT / "prompts/220-tag-prompts.csv",
        "test_prompt_column": "en_prompt",
        "suite_prompt_column": "en_prompt",
        "extensions": {".mp4", ".webm", ".gif", ".avi", ".mov"},
    },
]

FORMAT_NAMES = {
    ".mp4": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
    ".mov": {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"},
    ".webm": {"matroska", "webm"},
    ".avi": {"avi"},
    ".gif": {"gif"},
    ".webp": {"webp", "webp_pipe"},
}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames or []


def add_issue(issues, dataset, issue_type, item, detail):
    issues.append(
        {
            "dataset": dataset,
            "type": issue_type,
            "item": str(item),
            "detail": str(detail),
        }
    )


def normalized_prompt(value):
    return str(value or "").strip()


class VideoInspector:
    """优先使用 FFmpeg；不可用时退回 Pillow/OpenCV。"""

    def __init__(self, timeout):
        self.timeout = timeout
        self.ffprobe = os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe")
        self.ffmpeg = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
        self.cv2 = None
        self.image_class = None
        self.image_sequence = None

        if self.ffprobe and self.ffmpeg:
            self.backend = "ffmpeg"
            return

        try:
            import cv2
            from PIL import Image, ImageSequence

            self.cv2 = cv2
            self.image_class = Image
            self.image_sequence = ImageSequence
            self.backend = "pillow+opencv"
        except Exception as exc:
            self.backend = "unavailable"
            self.backend_error = str(exc)

    def inspect(self, path):
        if self.backend == "ffmpeg":
            return self._inspect_ffmpeg(path)
        if self.backend == "pillow+opencv":
            return self._inspect_python(path)
        return {
            "format_ok": False,
            "format_detail": f"没有可用的视频检查后端: {self.backend_error}",
            "decode_ok": False,
            "decode_detail": "未执行损坏检查",
        }

    def _inspect_ffmpeg(self, path):
        try:
            probe = subprocess.run(
                [
                    self.ffprobe,
                    "-v", "error",
                    "-show_entries", "format=format_name:stream=codec_type,codec_name,width,height",
                    "-of", "json",
                    str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "format_ok": False,
                "format_detail": f"ffprobe 超时（{self.timeout} 秒）",
                "decode_ok": False,
                "decode_detail": "未执行完整解码",
            }

        if probe.returncode != 0:
            return {
                "format_ok": False,
                "format_detail": probe.stderr.strip() or "ffprobe 无法识别文件",
                "decode_ok": False,
                "decode_detail": "格式探测失败，未执行完整解码",
            }

        try:
            metadata = json.loads(probe.stdout)
        except json.JSONDecodeError as exc:
            return {
                "format_ok": False,
                "format_detail": f"ffprobe 返回无效 JSON: {exc}",
                "decode_ok": False,
                "decode_detail": "未执行完整解码",
            }

        streams = metadata.get("streams", [])
        has_video_stream = any(stream.get("codec_type") == "video" for stream in streams)
        format_name = metadata.get("format", {}).get("format_name", "")
        detected_names = {name.strip() for name in format_name.split(",") if name.strip()}
        expected_names = FORMAT_NAMES.get(path.suffix.lower(), set())
        container_matches = bool(detected_names & expected_names)
        format_ok = has_video_stream and container_matches
        if not format_ok:
            return {
                "format_ok": False,
                "format_detail": f"扩展名={path.suffix.lower()}，探测格式={format_name or 'unknown'}，视频流={has_video_stream}",
                "decode_ok": False,
                "decode_detail": "格式不合法，未执行完整解码",
            }

        try:
            decode = subprocess.run(
                [
                    self.ffmpeg,
                    "-v", "error",
                    "-xerror",
                    "-i", str(path),
                    "-map", "0:v:0",
                    "-f", "null",
                    "-",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "format_ok": True,
                "format_detail": f"格式={format_name}",
                "decode_ok": False,
                "decode_detail": f"完整解码超时（{self.timeout} 秒）",
            }

        return {
            "format_ok": True,
            "format_detail": f"格式={format_name}",
            "decode_ok": decode.returncode == 0,
            "decode_detail": decode.stderr.strip() if decode.returncode else "完整解码成功",
        }

    def _inspect_python(self, path):
        extension = path.suffix.lower()
        if extension in {".gif", ".webp"}:
            expected_format = extension[1:].upper()
            try:
                with self.image_class.open(path) as image:
                    actual_format = str(image.format or "").upper()
                    format_ok = actual_format == expected_format
                if not format_ok:
                    return {
                        "format_ok": False,
                        "format_detail": f"扩展名={extension}，Pillow 探测格式={actual_format or 'unknown'}",
                        "decode_ok": False,
                        "decode_detail": "格式不合法，未执行完整解码",
                    }
                frame_count = 0
                with self.image_class.open(path) as image:
                    for frame in self.image_sequence.Iterator(image):
                        frame.load()
                        frame_count += 1
                return {
                    "format_ok": True,
                    "format_detail": f"Pillow 格式={actual_format}",
                    "decode_ok": frame_count > 0,
                    "decode_detail": f"成功解码 {frame_count} 帧" if frame_count else "没有可解码帧",
                }
            except Exception as exc:
                return {
                    "format_ok": False,
                    "format_detail": f"Pillow 无法打开文件: {exc}",
                    "decode_ok": False,
                    "decode_detail": f"图片/动画解码失败: {exc}",
                }

        capture = self.cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            return {
                "format_ok": False,
                "format_detail": "OpenCV 无法打开视频容器",
                "decode_ok": False,
                "decode_detail": "未能开始解码",
            }

        expected_frames = int(capture.get(self.cv2.CAP_PROP_FRAME_COUNT) or 0)
        decoded_frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                break
            decoded_frames += 1
        capture.release()

        decode_ok = decoded_frames > 0
        if expected_frames > 0 and decoded_frames + 1 < expected_frames:
            decode_ok = False
        return {
            "format_ok": True,
            "format_detail": "OpenCV 可识别视频容器",
            "decode_ok": decode_ok,
            "decode_detail": f"报告帧数={expected_frames}，成功解码={decoded_frames}",
        }


def validate_dataset(spec, inspector, workers, issues):
    dataset = spec["name"]
    test_rows, test_columns = read_csv(spec["test_csv"])
    suite_rows, suite_columns = read_csv(spec["suite_csv"])

    required_test_columns = {"name", spec["test_prompt_column"]}
    if not required_test_columns.issubset(test_columns):
        add_issue(issues, dataset, "prompt_coverage", spec["test_csv"], f"测试 CSV 缺少列: {sorted(required_test_columns - set(test_columns))}")
        return {"test_rows": len(test_rows), "existing_videos": 0, "valid_formats": 0, "undamaged_videos": 0, "suite_prompts": 0, "covered_suite_prompts": 0}
    if spec["suite_prompt_column"] not in suite_columns:
        add_issue(issues, dataset, "prompt_coverage", spec["suite_csv"], f"prompt suite CSV 缺少列: {spec['suite_prompt_column']}")
        return {"test_rows": len(test_rows), "existing_videos": 0, "valid_formats": 0, "undamaged_videos": 0, "suite_prompts": 0, "covered_suite_prompts": 0}

    test_prompts = {normalized_prompt(row.get(spec["test_prompt_column"])) for row in test_rows if normalized_prompt(row.get(spec["test_prompt_column"]))}
    suite_prompts = {normalized_prompt(row.get(spec["suite_prompt_column"])) for row in suite_rows if normalized_prompt(row.get(spec["suite_prompt_column"]))}
    missing_prompts = sorted(suite_prompts - test_prompts)
    for prompt in missing_prompts:
        add_issue(issues, dataset, "prompt_coverage", prompt, "prompt suite 中的 prompt 未被测试 CSV 覆盖")

    existing_paths = []
    seen_paths = set()
    for row in test_rows:
        name = str(row.get("name") or "").strip()
        path = spec["video_dir"] / name
        if not name or not path.is_file():
            add_issue(issues, dataset, "missing_video_file", name or "<empty name>", f"文件不存在: {path}")
            continue
        if path not in seen_paths:
            existing_paths.append(path)
            seen_paths.add(path)

    valid_format_count = 0
    undamaged_count = 0
    inspect_paths = []
    for path in existing_paths:
        if path.suffix.lower() not in spec["extensions"]:
            add_issue(issues, dataset, "invalid_video_format", path.name, f"不支持的扩展名: {path.suffix.lower() or '<none>'}")
        else:
            inspect_paths.append(path)

    if inspect_paths:
        print(f"[{dataset}] 使用 {inspector.backend} 检查 {len(inspect_paths)} 个视频，workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(inspector.inspect, path): path for path in inspect_paths}
            for completed, future in enumerate(as_completed(futures), 1):
                path = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    add_issue(issues, dataset, "damaged_video", path.name, f"检查程序异常: {exc}")
                    continue
                if not result["format_ok"]:
                    add_issue(issues, dataset, "invalid_video_format", path.name, result["format_detail"])
                    continue
                valid_format_count += 1
                if not result["decode_ok"]:
                    add_issue(issues, dataset, "damaged_video", path.name, result["decode_detail"])
                    continue
                undamaged_count += 1
                if completed % 100 == 0 or completed == len(inspect_paths):
                    print(f"[{dataset}] 已检查 {completed}/{len(inspect_paths)}")

    return {
        "test_rows": len(test_rows),
        "existing_videos": len(existing_paths),
        "valid_formats": valid_format_count,
        "undamaged_videos": undamaged_count,
        "suite_prompts": len(suite_prompts),
        "covered_suite_prompts": len(suite_prompts) - len(missing_prompts),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="仅用于在报告中记录本次运行目录")
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, default=max(1, int(os.environ.get("VALIDATE_VIDEO_WORKERS", "4"))))
    parser.add_argument("--video-timeout", type=int, default=max(1, int(os.environ.get("VALIDATE_VIDEO_TIMEOUT", "300"))))
    args = parser.parse_args()

    issues = []
    inspector = VideoInspector(timeout=args.video_timeout)
    print(f"视频检查后端: {inspector.backend}")

    summary = {}
    for spec in DATASETS:
        summary[spec["name"]] = validate_dataset(spec, inspector, max(1, args.workers), issues)

    passed = not issues
    report = {
        "passed": passed,
        "run_dir": str(Path(args.run_dir).resolve()),
        "video_backend": inspector.backend,
        "summary": summary,
        "issue_count": len(issues),
        "issues": issues,
    }
    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n========== 输入完整性检查结果 ==========")
    for dataset, stats in summary.items():
        print(
            f"{dataset}: 文件存在 {stats['existing_videos']}/{stats['test_rows']}，"
            f"格式有效 {stats['valid_formats']}，未损坏 {stats['undamaged_videos']}，"
            f"prompt 覆盖 {stats['covered_suite_prompts']}/{stats['suite_prompts']}"
        )

    if passed:
        print(f"[PASS] 测试视频与 prompt suite 检查全部通过。报告: {report_path}")
        return 0

    print(f"[FAIL] 共发现 {len(issues)} 个问题。完整报告: {report_path}")
    for issue in issues[:MAX_TERMINAL_ISSUES]:
        print(f"  [{issue['dataset']}][{issue['type']}] {issue['item']} | {issue['detail']}")
    if len(issues) > MAX_TERMINAL_ISSUES:
        print(f"  ... 其余 {len(issues) - MAX_TERMINAL_ISSUES} 条请查看 JSON 报告")
    return 1


if __name__ == "__main__":
    sys.exit(main())
