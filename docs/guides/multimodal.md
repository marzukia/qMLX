# Multimodal Models (Images & Video)

vllm-mlx supports vision-language models for image and video understanding.

## Supported Models

- Qwen3-VL (recommended)
- Qwen2-VL
- LLaVA
- Idefics
- PaliGemma
- Pixtral
- Molmo
- DeepSeek-VL

## Starting a Multimodal Server

```bash
vllm-mlx serve mlx-community/Qwen3-VL-4B-Instruct-3bit --port 8000
```

Models with "VL", "Vision", or "mllm" in the name are auto-detected as multimodal.

## Image Analysis

### Via OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

# Image from URL
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }],
    max_tokens=256
)
print(response.choices[0].message.content)
```

### Base64 Images

```python
import base64

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

base64_image = encode_image("photo.jpg")
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]
    }]
)
```

### Via curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
      ]
    }],
    "max_tokens": 256
  }'
```

## Video Analysis

### Via OpenAI SDK

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What happens in this video?"},
            {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
        ]
    }],
    max_tokens=512
)
```

### Video Parameters

Control frame extraction via extra body parameters:

```python
response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this video"},
            {"type": "video_url", "video_url": {"url": "video.mp4"}}
        ]
    }],
    extra_body={
        "video_fps": 2.0,
        "video_max_frames": 32
    }
)
```

### Via curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this video"},
        {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}
      ]
    }],
    "video_fps": 2.0,
    "video_max_frames": 16
  }'
```

## Supported Formats

### Images

| Format | Example |
|--------|---------|
| URL | `{"type": "image_url", "image_url": {"url": "https://..."}}` |
| Local file | `{"type": "image_url", "image_url": {"url": "/path/to/image.jpg"}}` |
| Base64 | `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}` |

### Videos

| Format | Example |
|--------|---------|
| URL | `{"type": "video_url", "video_url": {"url": "https://..."}}` |
| Local file | `{"type": "video", "video": "/path/to/video.mp4"}` |
| Base64 | `{"type": "video_url", "video_url": {"url": "data:video/mp4;base64,..."}}` |

## Python API

```python
from vllm_mlx.models import MLXMultimodalLM

mllm = MLXMultimodalLM("mlx-community/Qwen3-VL-4B-Instruct-3bit")
mllm.load()

# Image
description = mllm.describe_image("photo.jpg")

# Video
description = mllm.describe_video("video.mp4", fps=2.0)

# Custom prompt
output = mllm.generate(
    prompt="Compare these images",
    images=["img1.jpg", "img2.jpg"]
)
```

## Performance Tips

### Images
- Smaller resolutions process faster (224x224 vs 1920x1080)
- Use appropriate resolution for your task

### Videos
- Lower FPS = faster processing
- Fewer frames = less memory usage
- 64 frames is practical maximum (96+ causes GPU timeout)

## Gradio Chat UI

For interactive multimodal chat:

```bash
vllm-mlx-chat --model mlx-community/Qwen3-VL-4B-Instruct-3bit
```

Supports drag-and-drop images and videos.
