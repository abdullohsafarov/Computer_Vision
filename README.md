# CV LLM — Natural Language Image Processor

Control OpenCV with plain text commands via a local LLM (LM Studio).

## How it works

1. You type a command in any language (e.g. *"rotate 90 degrees"*)
2. The command is sent to a local LM Studio model
3. The LLM returns a structured JSON: `{"op": "rotate", "angle": 90}`
4. OpenCV executes the operation and saves the result to `output.jpg`

## Requirements

- Python 3.8+
- [LM Studio](https://lmstudio.ai) running locally with a model loaded and the local server enabled

## Setup

```bash
pip install -r requirements.txt
```

Edit `.env` if needed:

```
LM_STUDIO_URL=http://localhost:1234/v1/chat/completions
MODEL_NAME=local-model
```

## Usage

```bash
python main.py <image_path>
```

Then type commands interactively. Type `exit` to quit.

## Supported Operations

| Command example | Operation | Parameters |
|---|---|---|
| `rotate 90 degrees` | Rotate image | `angle` (int, degrees) |
| `resize to 800x600` | Resize image | `width`, `height` (px) |
| `show only red channel` | Isolate red channel | — |
| `convert to grayscale` | Grayscale | — |
| `blur with kernel 15` | Gaussian blur | `kernel` (odd int) |

## Project Structure

```
cv_llm/
├── main.py          # entry point: LLM routing + OpenCV execution
├── .env             # LM Studio URL and model name
└── requirements.txt
```
