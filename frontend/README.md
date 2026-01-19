# SEAL Frontend: Streamlit Visualization Interface

This is a **read-only** frontend for the SEAL (Self-Edit Adaptive Learning) continual learning research project.

## Features

The frontend provides 5 interactive screens:

1. **Overview**: Project background and SEAL methodology explanation
2. **Accuracy Matrix Viewer**: Visualize pre-computed accuracy metrics from `outputs/multi_task/`
3. **Method Comparison**: Side-by-side comparison of different techniques (baseline, SEAL, Hybrid, EWC)
4. **Forgetting Analysis**: Detailed analysis of catastrophic forgetting and how SEAL addresses it
5. **Chatbot**: LLM-powered Q&A using llama2 via Ollama (explanatory only)

## Installation

### Prerequisites

- Python 3.8+
- The SEAL backend already installed and run (to generate `outputs/multi_task/*.json`)
- Ollama (optional, for chatbot feature): [https://ollama.ai](https://ollama.ai)

### Setup

```bash
# Install dependencies
pip install -r requirements_frontend.txt

# (Optional) Start Ollama
ollama serve

# (Optional) Pull llama2 model
ollama pull llama2
```

## Running the Frontend

```bash
streamlit run frontend/app.py
```

The app will open at `http://localhost:8501`.

## Data Sources

- **Accuracy matrices**: `outputs/multi_task/hybrid/imdb_squad_arc_metrics.json`
- **Task results**: `outputs/multi_task/hybrid/task_results.json`
- **Screenshots**: `frontend/assets/screenshots/` (place method comparison images here)

## Screenshots Directory Structure

Place comparison images in `frontend/assets/screenshots/`:

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

## Chatbot Configuration

The chatbot uses **Ollama** for LLM inference:

- **Model**: llama2 (configurable in `frontend/app.py`)
- **Endpoint**: `http://localhost:11434`
- **Cloud Deployment**: The chatbot requires local Ollama and is **expected to be unavailable** in cloud deployments (Streamlit Cloud, etc.)
- **Fallback**: If Ollama is unavailable, the chatbot gracefully disables with an informative message

To enable the chatbot **locally**:

```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Pull llama2
ollama pull llama2

# Terminal 3: Run frontend
streamlit run frontend/app.py
```

## Architecture

```
frontend/
├── app.py                      # Main Streamlit app (5 screens)
├── utils.py                    # Utility functions
├── requirements_frontend.txt   # Python dependencies
├── assets/
│   └── screenshots/            # Method comparison images
└── README.md                   # This file
```

## Key Design Principles

✅ **Read-Only**: Frontend never modifies backend data or models
✅ **Standalone**: Works independently without modifying backend code
✅ **Graceful Degradation**: Missing data/Ollama handled with informative messages
✅ **Academic Layout**: Clean, professional presentation suitable for research contexts
✅ **No Training Buttons**: All training/modification happens offline in backend

## Constraints

- ❌ No training or model modification
- ❌ No FastAPI or backend integration
- ❌ No database or persistent storage (read-only from JSON files)
- ❌ No React or JavaScript frameworks (Streamlit only)

## Troubleshooting

### Chatbot not working?
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start Ollama
ollama serve
```

### No accuracy matrix visible?
- Ensure the backend has been run: `python main.py --mode tasks`
- Check that `outputs/multi_task/hybrid/imdb_squad_arc_metrics.json` exists

### Screenshots not showing?
- Place PNG files in `frontend/assets/screenshots/`
- Use the exact filenames from `app.py` (e.g., `baseline.png`, `hybrid_ewc_final.png`)

## License

This frontend is part of the SEAL research project. See the main project LICENSE.
