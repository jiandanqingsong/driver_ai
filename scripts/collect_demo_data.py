from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CLASS_LABELS = {
    "c0": "safe_driving",
    "c1": "texting_right",
    "c2": "talking_phone_right",
    "c3": "texting_left",
    "c4": "talking_phone_left",
    "c5": "operating_radio",
    "c6": "drinking",
    "c7": "reaching_behind",
    "c8": "hair_and_makeup",
    "c9": "talking_to_passenger",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect demo-scene images from a PC camera.")
    parser.add_argument("--source", default="0", help="Camera index or video source. Default: 0.")
    parser.add_argument("--output-root", default="data/demo_scene")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between saved images.")
    parser.add_argument("--duration", type=float, default=100.0, help="Capture duration after warmup.")
    parser.add_argument("--warmup", type=float, default=5.0, help="Countdown seconds after clicking start.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--class-id", choices=sorted(CLASS_LABELS), default=None, help="Skip GUI and record this class.")
    parser.add_argument("--no-voice", action="store_true")
    return parser.parse_args()


def speak(message: str, enabled: bool = True) -> None:
    print(message)
    if not enabled:
        return
    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(message)
        engine.runAndWait()
    except Exception as exc:
        print(f"Voice prompt skipped: {exc}")


def choose_class_gui() -> str:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception:
        return choose_class_cli()

    selected = {"value": "c0"}

    root = tk.Tk()
    root.title("选择采集分类")
    root.geometry("420x360")
    root.resizable(False, False)

    title = ttk.Label(root, text="请选择本次录制的分类 C0-C9", font=("Microsoft YaHei UI", 12, "bold"))
    title.pack(pady=(16, 8))

    class_var = tk.StringVar(value="c0")
    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True, padx=24)

    for class_id, name in CLASS_LABELS.items():
        ttk.Radiobutton(frame, text=f"{class_id.upper()}  {name}", value=class_id, variable=class_var).pack(
            anchor="w",
            pady=2,
        )

    def start() -> None:
        selected["value"] = class_var.get()
        root.destroy()

    button = ttk.Button(root, text="开始捕获", command=start)
    button.pack(pady=(8, 18))
    root.mainloop()
    return selected["value"]


def choose_class_cli() -> str:
    print("请选择本次录制的分类：")
    for class_id, name in CLASS_LABELS.items():
        print(f"  {class_id.upper()}: {name}")
    while True:
        value = input("输入 C0-C9: ").strip().lower()
        if value in CLASS_LABELS:
            return value
        print("输入无效，请重新输入。")


def next_index(output_dir: Path, class_id: str) -> int:
    pattern = re.compile(rf"^{re.escape(class_id)}_(\d+)\.jpg$", re.IGNORECASE)
    max_index = 0
    if output_dir.exists():
        for path in output_dir.glob(f"{class_id}_*.jpg"):
            match = pattern.match(path.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def open_capture(source: str, width: int, height: int):
    video_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(video_source)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera/video source: {source}")
    return cap


def draw_status(frame, lines: list[str]):
    panel_height = 92
    cv2.rectangle(frame, (0, 0), (frame.shape[1], panel_height), (20, 20, 20), -1)
    for idx, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (18, 30 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def run_collection(args: argparse.Namespace) -> None:
    class_id = args.class_id or choose_class_gui()
    class_name = CLASS_LABELS[class_id]
    output_dir = Path(args.output_root) / class_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = open_capture(args.source, args.width, args.height)
    window_name = f"Collect {class_id.upper()} - {class_name}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    speak(f"Selected {class_id.upper()}, {class_name}. Capture will start after {int(args.warmup)} seconds.", not args.no_voice)

    start_time = time.time()
    capture_start = start_time + args.warmup
    capture_end = capture_start + args.duration
    next_capture_time = capture_start
    file_index = next_index(output_dir, class_id)
    saved = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            display = frame.copy()

            if now < capture_start:
                remain = max(0.0, capture_start - now)
                draw_status(
                    display,
                    [
                        f"Class: {class_id.upper()} {class_name}",
                        f"Starting in {remain:.1f}s",
                        "Press Q or ESC to stop",
                    ],
                )
            else:
                if now >= next_capture_time and now <= capture_end:
                    filename = f"{class_id}_{file_index:04d}.jpg"
                    save_path = output_dir / filename
                    cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, int(args.quality)])
                    file_index += 1
                    saved += 1
                    next_capture_time += args.interval

                remain = max(0.0, capture_end - now)
                draw_status(
                    display,
                    [
                        f"Class: {class_id.upper()} {class_name}",
                        f"Saved: {saved}  Remaining: {remain:.1f}s",
                        f"Output: {output_dir}",
                    ],
                )

                if now >= capture_end:
                    break

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q"), ord("Q")}:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    speak(f"Capture finished. Saved {saved} images for {class_id.upper()}.", not args.no_voice)
    print(f"Saved {saved} images to {output_dir}")


def main() -> None:
    args = parse_args()
    run_collection(args)


if __name__ == "__main__":
    main()
