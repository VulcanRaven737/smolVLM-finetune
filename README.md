---
language:
- en
license: apache-2.0
tags:
- multimodal
- vision-language
- smolvlm
- lora
- peft
- chart-qa
- document-understanding
datasets:
- HuggingFaceM4/ChartQA
base_model:
- HuggingFaceTB/SmolVLM-500M-Instruct
pipeline_tag: image-text-to-text
---

# ChartQA-smolvlm

SmolVLM-500M-Instruct fine-tuned with LoRA on [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) for chart question answering. LoRA adapters were merged into the base model for single-artifact deployment.

## Model details

| | |
|---|---|
| **Base model** | `HuggingFaceTB/SmolVLM-500M-Instruct` |
| **Dataset** | `HuggingFaceM4/ChartQA` |
| **Task** | Visual question answering on chart/graph images |
| **Method** | LoRA (`r=16`, `α=32`, all projection layers, dropout=0.05) |
| **Target modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Precision** | bfloat16 |
| **Epochs** | 3 |
| **Batch size** | 4 |
| **LR** | 2e-4 with cosine warmup (5%) |
| **Optimizer** | AdamW (fused) |
| **Metrics** | Exact Match, Relaxed Accuracy (5% tol), ANLS |

## Usage

### Full model (recommended)

```python
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

model_id = "VulcanRaven/ChartQA-smolvlm"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
).cuda().eval()

image = Image.open("chart.png").convert("RGB")
query = "What is the highest value shown in the chart?"

messages = [{
    "role": "user",
    "content": [
        {"type": "image"},
        {"type": "text", "text": f"Question: {query}\nAnswer:"}
    ]
}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=text, images=image, return_tensors="pt").to("cuda")
inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

with torch.inference_mode():
    gen_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
answer = processor.tokenizer.decode(
    gen_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
).strip()
print(f"Q: {query}\nA: {answer}")
```

### LoRA adapter variant (load + merge before inference)

```python
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch

base = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM-500M-Instruct", torch_dtype=torch.bfloat16
)
model = PeftModel.from_pretrained(base, "VulcanRaven/ChartQA-smolvlm")
model = model.merge_and_unload().cuda().eval()
processor = AutoProcessor.from_pretrained("VulcanRaven/ChartQA-smolvlm")

# then run inference as above
```

## Evaluation metrics

| Metric | Description |
|---|---|
| **Exact Match** | Normalised string equality against any gold answer |
| **Relaxed Accuracy** | Numeric tolerance of ±5%; falls back to exact match for non-numeric answers |
| **ANLS** | Average Normalised Levenshtein Similarity (threshold=0.5) |

## Design decisions

| Decision | Choice | Reason |
|---|---|---|
| Base model | SmolVLM-500M-Instruct | Compact VLM with strong chart understanding; fits in <4 GB VRAM |
| Dataset | ChartQA | Standard benchmark for chart visual QA with multi-reference gold answers |
| Fine-tuning | LoRA on all projection layers | Covers attention + MLP; fast convergence with minimal memory overhead |
| Label masking | Prefix tokens masked to -100 | Model only learns to generate the answer, not repeat the question |
| Deployment | Merged full model | No adapter loading code at inference; simpler and faster |
| Precision | bfloat16 | Numerically stable; works well even on resource-constrained GPUs |

## Training details

- **Hardware**: NVIDIA Tesla T4
- **Data split**: 80% train / 20% validation (from original train), full original test set
- **Validation**: 50-batch subset evaluated after each epoch for speed
- **Best checkpoint**: Saved based on highest Relaxed Accuracy on validation set
- **Gradient clipping**: Max norm 1.0
- **Grad accumulation**: 4 steps (effective batch size 16)
