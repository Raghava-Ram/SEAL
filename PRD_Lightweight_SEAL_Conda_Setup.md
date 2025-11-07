# Lightweight SEAL - Conda + CPU-Compatible Setup

## 📝 PRD (Product Requirements Document)

### 🎯 Objective
Modify the SEAL repository to:
- Run locally on CPU-only using DistilBERT
- Keep the self-adaptive SEAL loop intact
- Simulate OpenAI edits locally
- Structure code for easy OpenAI API integration
- Use Conda for environment and dependency management

## 🖥️ Target System

| Component | Value |
|-----------|-------|
| CPU | AMD Ryzen 5 5600H |
| GPU | Radeon (no CUDA) |
| OS | Windows/Linux |
| RAM | 8-16 GB |
| Environment | Conda (Anaconda/Miniconda) |

## 🛠️ Setup Instructions

### 1. Environment Setup (Conda-Based)

```bash
# Create and activate conda environment
conda create -n seal_env python=3.10 -y
conda activate seal_env

# Install CPU-only PyTorch
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### 2. Required Code Modifications

#### `main.py` (or model initialization)
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
device = torch.device("cpu")
model.to(device)
```

#### `seal/adapter.py`
```python
def simulate_edit_locally(text: str) -> str:
    """Simulate GPT-style edit locally (offline version)."""
    return text.replace("is", "might be").replace("the", "this")

def generate_edit(text: str, mode="local") -> str:
    """Switch between local or OpenAI-based editing."""
    if mode == "local":
        return simulate_edit_locally(text)
    elif mode == "openai":
        from seal.openai_edit import generate_edit_via_openai
        return generate_edit_via_openai(text)
```

#### `seal/openai_edit.py` (future use)
```python
# Placeholder for GPT-based editing (not used in local mode)
import openai
import os

def generate_edit_via_openai(text: str) -> str:
    """Use OpenAI GPT model to generate edits (future phase)."""
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Improve factual accuracy and clarity."},
            {"role": "user", "content": f"Edit this: {text}"}
        ]
    )
    return response.choices[0].message["content"].strip()
```

#### `configs/default.yaml`
```yaml
edit_mode: "local"         # Switch to "openai" later
batch_size: 1
max_steps: 10
device: "cpu"
model_name: "distilbert-base-uncased"
```

## 🧪 Testing

### `test_seal_cpu.py`
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
from seal.adapter import generate_edit

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.to("cpu")

text = "Machine learning is transforming the world."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

edit = generate_edit(text, mode="local")
print("✅ Model ran successfully on CPU")
print("Generated Edit:", edit)
```

Run the test:
```bash
python test_seal_cpu.py
```

## ✅ Validation Criteria
- [ ] Environment builds cleanly via Conda
- [ ] DistilBERT loads and runs on CPU
- [ ] No GPU references remain
- [ ] Local edit simulation works
- [ ] Config supports edit_mode toggle
- [ ] SEAL loop runs offline
- [ ] Test script executes successfully

## 🚀 Future Enablement (OpenAI Mode)
1. Set your API key:
   ```bash
   setx OPENAI_API_KEY "your-key"
   ```
2. Update config:
   ```yaml
   edit_mode: "openai"
   ```

## 📊 Summary

| Component | Change | Purpose |
|-----------|--------|----------|
| Environment | Conda + CPU PyTorch | Stability, reproducibility |
| Model | DistilBERT | Lightweight |
| OpenAI API | Modular, optional | Future upgrade-ready |
| Self-Adaptive Loop | Preserved | Research authenticity |
| Config | Simplified | Easier debugging |
| Testing | Added | Quick validation |

## 🔄 Workflow Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  Input Text    │────>│  SEAL Loop      │────>│  Local Edit      │
└─────────────────┘     │  (CPU)          │     │  Simulation      │
                        └────────┬────────┘     └────────┬─────────┘
                                 │                       │
                                 ▼                       ▼
                        ┌─────────────────┐     ┌──────────────────┐
                        │  Model          │     │  Output Text     │
                        │  (DistilBERT)   │<────│  with Edits      │
                        └─────────────────┘     └──────────────────┘
```

## 📝 Notes
- The system is designed to work offline by default
- OpenAI integration is modular and can be easily enabled
- All components are designed to be lightweight and run on CPU
- The configuration allows for easy toggling between local and OpenAI modes
