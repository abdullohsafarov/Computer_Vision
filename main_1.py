import cv2
import requests
import json
import os
import sys
import numpy as np
from dotenv import load_dotenv

load_dotenv()

LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1/chat/completions")
MODEL_NAME    = os.getenv("MODEL_NAME", "local-model")

SYSTEM_PROMPT = """You are an image processing assistant. Parse the user's command and return ONLY a JSON object (no markdown, no explanation).

Supported operations and their JSON format:

1. rotate     → {"op": "rotate", "angle": <int>}
2. resize     → {"op": "resize", "width": <int>, "height": <int>}
3. red_channel → {"op": "red_channel"}
4. grayscale  → {"op": "grayscale"}
5. blur       → {"op": "blur", "kernel": <odd_int>}

Rules:
- Return ONLY the JSON object.
- If the command is unclear, return {"op": "unknown"}.
- For rotate, extract the angle in degrees (e.g. "90", "180", "-45").
- For blur, kernel must be an odd integer (e.g. 5, 7, 15). Default: 5.
- For resize, extract width and height in pixels.
"""


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
        # strip possible markdown fences
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"[LLM error] {e}")
        return {"op": "unknown"}


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
        result[:, :, 2] = img[:, :, 2]   # OpenCV: BGR, so red = channel 2
        return result

    elif op == "grayscale":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # keep 3 channels for saving

    elif op == "blur":
        k = cmd.get("kernel", 5)
        if k % 2 == 0:
            k += 1                         # kernel must be odd
        return cv2.GaussianBlur(img, (k, k), 0)

    else:
        print(f"[!] Unknown operation: {op}")
        return img


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    img = cv2.imread(image_path)
    if img is None:
        print(f"[!] Cannot open image: {image_path}")
        sys.exit(1)

    print(f"[+] Loaded: {image_path}  ({img.shape[1]}x{img.shape[0]})")
    print("Supported commands: rotate, resize, red channel, grayscale, blur")
    print("Type 'exit' to quit.\n")

    current = img.copy()

    while True:
        user_input = input("Command: ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue

        print("[*] Sending to LLM...")
        cmd = parse_command(user_input)
        print(f"[*] Parsed command: {cmd}")

        if cmd.get("op") == "unknown":
            print("[!] Could not understand the command.\n")
            continue

        current = apply_operation(current, cmd)

        out_path = "output.jpg"
        cv2.imwrite(out_path, current)
        print(f"[+] Saved → {out_path}\n")

        cv2.imshow("Result", current)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    print("Bye.")


if __name__ == "__main__":
    main()
