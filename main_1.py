import cv2
import requests
import json
import os
import re
import sys
import numpy as np
from dotenv import load_dotenv
from tkinter import filedialog, simpledialog, messagebox, Tk
import tkinter as tk

load_dotenv()

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")
MODEL_NAME    = os.getenv("MODEL_NAME", "llava-v1.6-mistral-7b:2")

SYSTEM_PROMPT = """You are an image processing assistant. Parse the user's command and return ONLY a JSON object — no markdown, no explanation, no extra text.

Supported operations:

1. rotate      → {"op": "rotate", "angle": <int>}
2. resize      → {"op": "resize", "width": <int>, "height": <int>}
3. red_channel → {"op": "red_channel"}
4. grayscale   → {"op": "grayscale"}
5. blur        → {"op": "blur", "kernel": <odd_int>}

Defaults when a parameter is missing:
- rotate with no angle → use 90
- blur with no kernel  → use 5
- resize with no size  → use width 640, height 480

Keywords to recognize (any language):
- rotate / поверни / повернуть / вращать          → rotate
- resize / измени размер / масштаб / разрешение   → resize
- red / красный / красная компонента              → red_channel
- gray / grey / серый / чёрно-белый / grayscale   → grayscale
- blur / размытие / размыть / блур                → blur

Rules:
- Return ONLY the JSON object, nothing else.
- If you cannot match any operation, return {"op": "unknown"}.
"""


def keyword_fallback(text: str) -> dict:
    """Simple keyword matcher as fallback when LLM fails or returns unknown."""
    t = text.lower()

    if any(w in t for w in ("rotate", "поверн", "вращ")):
        nums = re.findall(r"-?\d+", t)
        angle = int(nums[0]) if nums else 90
        return {"op": "rotate", "angle": angle}

    if any(w in t for w in ("resize", "размер", "масштаб", "разрешени")):
        nums = re.findall(r"\d+", t)
        w, h = (int(nums[0]), int(nums[1])) if len(nums) >= 2 else (640, 480)
        return {"op": "resize", "width": w, "height": h}

    if any(w in t for w in ("red", "красн")):
        return {"op": "red_channel"}

    if any(w in t for w in ("gray", "grey", "серый", "чёрно", "черно", "grayscale")):
        return {"op": "grayscale"}

    if any(w in t for w in ("blur", "размыт", "блур")):
        nums = re.findall(r"\d+", t)
        k = int(nums[0]) if nums else 5
        return {"op": "blur", "kernel": k}

    return {"op": "unknown"}


def parse_command(user_input: str) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_input},
        ],
        "temperature": 0.0,
    }
    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=30)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        text = text.replace("```json", "").replace("```", "").strip()
        result = json.loads(text)
        if result.get("op") == "unknown":
            return keyword_fallback(user_input)
        return result
    except Exception as e:
        print(f"[LLM error] {e}")
        return keyword_fallback(user_input)


def apply_operation(img: np.ndarray, cmd: dict) -> np.ndarray:
    op = cmd.get("op", "unknown")

    if op == "rotate":
        angle = cmd.get("angle", 90)
        h, w  = img.shape[:2]
        center = (w // 2, h // 2)
        M      = cv2.getRotationMatrix2D(center, -angle, 1.0)
        return cv2.warpAffine(img, M, (w, h))

    elif op == "resize":
        w = cmd.get("width", 640)
        h = cmd.get("height", 480)
        return cv2.resize(img, (w, h))

    elif op == "red_channel":
        result = np.zeros_like(img)
        result[:, :, 2] = img[:, :, 2]   # OpenCV BGR: red = channel 2
        return result

    elif op == "grayscale":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif op == "blur":
        k = cmd.get("kernel", 5)
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(img, (k, k), 0)

    else:
        print(f"[!] Unknown operation: {op}")
        return img


def main():
    # Create hidden root window for file dialog
    root = Tk()
    root.withdraw()
    
    # Open file picker dialog
    image_path = filedialog.askopenfilename(
        title="Выберите фото / Select Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
    )
    
    if not image_path:
        messagebox.showwarning("Info", "No image selected / Фото не выбрано")
        return
    
    img = cv2.imread(image_path)
    if img is None:
        messagebox.showerror("Error", f"Cannot open image / Не могу открыть: {image_path}")
        return
    
    messagebox.showinfo("Success", f"Loaded / Загружено:\n{image_path}\n({img.shape[1]}x{img.shape[0]})")
    
    print(f"[+] Loaded: {image_path}  ({img.shape[1]}x{img.shape[0]})")
    print("Commands: rotate [angle], resize [WxH], red channel, grayscale, blur [kernel]")
    print("Type 'exit' to quit.\n")

    current = img.copy()
    cv2.imshow("Original Image", img)

    while True:
        user_input = simpledialog.askstring(
            "Image Processing",
            "Введите команду / Enter command:\nrotate, resize, red channel, grayscale, blur\n(or 'exit' to quit)"
        )
        
        if user_input is None:  # Cancel button
            break
            
        user_input = user_input.strip()
        
        if user_input.lower() in ("exit", "quit", "q"):
            break
            
        if not user_input:
            continue

        print("[*] Sending to LLM...")
        cmd = parse_command(user_input)
        print(f"[*] Parsed: {cmd}")

        if cmd.get("op") == "unknown":
            messagebox.showwarning("Warning", "Could not understand command / Не понял команду")
            print("[!] Could not understand the command.\n")
            continue

        current = apply_operation(current, cmd)

        out_path = os.path.join(os.path.dirname(image_path), "output.jpg")
        cv2.imwrite(out_path, current)
        messagebox.showinfo("Success", f"Saved / Сохранено:\n{out_path}")
        print(f"[+] Saved → {out_path}\n")

        cv2.imshow("Result", current)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    root.destroy()
    print("Bye / До свидания.")


if __name__ == "__main__":
    main()
