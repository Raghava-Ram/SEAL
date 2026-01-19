# 🚀 SEAL Frontend Quick Start Guide

## What is This?

The **SEAL Frontend** is a read-only web interface for visualizing continual learning results from the SEAL (Self-Edit Adaptive Learning) research project. It displays pre-computed results, explains methods, and provides an LLM-powered chatbot for questions.

## 5 Screens Overview

| Screen | Purpose |
|--------|---------|
| **📖 Overview** | Explain catastrophic forgetting and SEAL methodology |
| **📊 Accuracy Matrix** | Visualize task accuracy over time with graphs |
| **🎯 Method Comparison** | Side-by-side screenshots of different techniques |
| **📉 Forgetting Analysis** | Deep dive into forgetting metrics and solutions |
| **💬 Chatbot** | Ask questions using llama2 LLM |

## Installation (One-Time)

### Step 1: Install Dependencies

```bash
pip install -r frontend/requirements_frontend.txt
```

### Step 2 (Optional): Setup Ollama for Chatbot

If you want the chatbot feature:

```bash
# Download and install Ollama from https://ollama.ai
# Then pull the llama2 model
ollama pull llama2
```

## Running the Frontend

### Option A: Simple Launch

```bash
streamlit run frontend/app.py
```

### Option B: Using Launch Script

```bash
python frontend/launch.py
```

This script checks dependencies and provides status messages.

The app will open at: **http://localhost:8501**

## Data Requirements

Before running the frontend, ensure the backend has generated results:

```bash
# Run the SEAL backend to generate outputs
python main.py --mode tasks
```

This creates:
- `outputs/multi_task/hybrid/imdb_squad_arc_metrics.json` (required)
- `outputs/multi_task/hybrid/task_results.json` (optional)

## Optional: Add Method Comparison Screenshots

The **Method Comparison** screen displays 7 images showing technique progression. To enable this:

1. Generate or collect PNG images for each method
2. Place them in `frontend/assets/screenshots/`:

```
frontend/assets/screenshots/
├── baseline.png
├── seal_replay.png
├── hybrid_llm_replay.png
├── hybrid_freezing.png
├── hybrid_task_weighted_replay.png
├── hybrid_task_weighted_replay_v2.png
└── hybrid_ewc_final.png
```

If images are missing, the screen shows a placeholder message—no error occurs.

## Chatbot Setup (Optional)

The chatbot uses **llama2 via Ollama**:

### Terminal 1: Start Ollama Server

```bash
ollama serve
```

### Terminal 2: Pull Model (first time only)

```bash
ollama pull llama2
```

### Terminal 3: Run Frontend

```bash
streamlit run frontend/app.py
```

Now the **Chatbot** screen will be fully functional!

### Chatbot Fallback

If Ollama is not running:
- Chatbot screen shows an error message with setup instructions
- Other screens remain fully functional
- No data loss or errors

## Troubleshooting

### "Metrics file not found"

```bash
# Backend hasn't been run yet
python main.py --mode tasks

# Or verify the file exists
ls outputs/multi_task/hybrid/imdb_squad_arc_metrics.json
```

### "Ollama is not running"

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull the model
ollama pull llama2

# Terminal 3: Frontend should now work
streamlit run frontend/app.py
```

### Frontend crashes or imports fail

```bash
# Reinstall dependencies
pip install --upgrade -r frontend/requirements_frontend.txt

# Run the test script
python frontend/test_frontend.py
```

### Screenshots not showing

- Check filenames match exactly: `baseline.png`, `hybrid_ewc_final.png`, etc.
- Ensure they're in `frontend/assets/screenshots/`
- PNG format required (not JPG or other formats)

## Architecture

```
frontend/
├── app.py                      # Main Streamlit app (ALL 5 screens)
├── utils.py                    # Shared utilities
├── launch.py                   # Launcher script
├── test_frontend.py            # Test script
├── requirements_frontend.txt   # Dependencies
├── README.md                   # Full documentation
├── assets/
│   └── screenshots/            # Method comparison images (optional)
└── __init__.py                 # Package marker
```

## Features

✅ **Read-Only**: No model training or data modification
✅ **Offline-Capable**: Works with pre-computed JSON files
✅ **Graceful Fallbacks**: Missing data/Ollama handled gracefully
✅ **Academic Design**: Clean, professional layout
✅ **No External APIs**: Only uses local Ollama (optional)
✅ **Standalone**: Doesn't modify backend code

## Project Structure

```
SEAL/
├── main.py                         # Backend entry point
├── requirements.txt                # Backend dependencies
├── frontend/                       # ← Frontend (NEW)
│   ├── app.py                      # Main Streamlit app
│   ├── utils.py                    # Utilities
│   ├── launch.py                   # Launcher
│   ├── test_frontend.py            # Tests
│   ├── requirements_frontend.txt   # Dependencies
│   ├── README.md                   # Full docs
│   ├── assets/
│   │   └── screenshots/            # Images
│   └── __init__.py
├── outputs/
│   └── multi_task/
│       └── hybrid/
│           ├── imdb_squad_arc_metrics.json
│           └── task_results.json
├── seal/                           # Backend code
└── ...
```

## Typical Workflow

1. **Run backend**: `python main.py --mode tasks`
2. **Prepare images** (optional): Add screenshots to `frontend/assets/screenshots/`
3. **Start Ollama** (optional): `ollama serve` (new terminal)
4. **Launch frontend**: `streamlit run frontend/app.py` or `python frontend/launch.py`
5. **Explore**: Navigate 5 screens, ask chatbot questions

## Support

For issues, refer to:
- [frontend/README.md](README.md) - Detailed documentation
- [frontend/test_frontend.py](test_frontend.py) - Diagnostic test
- Run `python frontend/test_frontend.py` to check system status

---

**Ready to explore SEAL results?** 🚀

```bash
streamlit run frontend/app.py
```
