# 🚀 SEAL-Lite: Lightweight Continual Learning Framework

A Final-Year Project on Overcoming Catastrophic Forgetting Using Adaptive Editing, Replay Memory, and Utility-Guided Learning

## 📘 Overview
SEAL-Lite is a lightweight continual learning system inspired by the research paper: Self-Edit Accumulation Learning (SEAL) (2025).

Traditional neural networks forget previously learned tasks when trained sequentially — a phenomenon called catastrophic forgetting. SEAL-Lite aims to solve this problem using:
- Adaptive edits (local or LLM-generated)
- Utility-based scoring
- Priority replay memory
- Self-supervised continual fine-tuning

The framework is fully CPU-compatible, uses DistilBERT, and supports local LLMs via Ollama.

## 🎯 Live Demo

**Frontend Demo (Read-Only Visualization)**  
▶️ [https://raghava-ram-seal-frontendapp-dandwk.streamlit.app/](https://raghava-ram-seal-frontendapp-dandwk.streamlit.app/)

Explore interactive visualizations of all 7 continual learning methods with accuracy heatmaps, forgetting analysis, and comparative results. Note: Chatbot requires local Ollama and is unavailable in cloud deployment.

## 📋 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [How It Works](#-how-it-works)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## 🧠 Key Features

### ✔ Continual Learning Engine
- Based on SEAL's edit → score → store → replay cycle
- Prevents catastrophic forgetting with replay + utility signals

### ✔ Dual Edit Modes
- Local rule-based editor (fast, offline)
- LLM-powered editor via Ollama (optional)

### ✔ Replay Memory
- JSONL-backed memory
- Priority sampling
- Stores high-utility edits

### ✔ Utility Scoring
- Measures how much each edit improves model confidence
- Ensures valuable edits stay in memory

### ✔ Trainer
- Batch training
- Confidence-based predictions
- CPU-only support

### ✔ Modular Architecture
Clear separation of:
- memory
- utility
- replay
- trainer
- prompt adapter
- LLM adapter
- runner

## 🛠 Installation

1. **Create environment**
   ```bash
   conda create -n seal_env python=3.10
   conda activate seal_env
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Run Ollama**
   ```bash
   ollama serve
   ```
   Pull a model:
   ```bash
   ollama pull llama2
   ```

## 🏁 Running the Project

### SEAL Continual Learning Loop
```bash
python main.py --mode seal
```
Runs 1000+ continual updates with memory + replay.

### IMDB Local vs LLM Comparison
```bash
python main.py --mode imdb
```
Shows performance difference between local edit mode and LLM edit mode.

### Lightweight Demonstration
```bash
python main.py --mode demo
```
Runs SEAL on 3 IMDB samples.

### Future: Multi-Task Sequential Training
```bash
python main.py --mode tasks
```
(Placeholder for IMDB → SQuAD → ARC sequence)

## 📂 Project Structure

```
SEAL/
│── main.py
│── configs/
│     └── default.yaml
│── seal/
│     ├── runner.py
│     ├── trainer.py
│     ├── adapter.py
│     ├── llm_adapter.py
│     ├── prompt_adapter.py
│     ├── memory.py
│     ├── replay.py
│     ├── utility.py
│     └── demo_seal.py
│── memory/
│     └── edits.jsonl
│── outputs/
│     ├── plots
│     └── results
└── README.md
```

## ⚙️ Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Model configuration
model:
  name: distilbert-base-uncased
  num_labels: 2

# Training parameters
trainer:
  learning_rate: 1e-4
  max_steps: 1000
  eval_interval: 10
  device: cpu  # or cuda

# Memory and replay
memory:
  path: "memory/edits.jsonl"
  max_size: 20000

replay:
  batch_size: 8
  replay_fraction: 0.5
  policy: "priority"  # or "uniform"

# LLM configuration (for LLM editing mode)
llm:
  backend: "ollama"  # or "openai"
  model: "llama2"
  timeout: 120
```

## ⚡ How SEAL-Lite Works

1. **Take a sample**
   From IMDB, SQuAD (future), or ARC (future).

2. **Generate an edit**
   - Local rule-based sentiment edit
   - OR LLM edit (via Ollama)

3. **Score utility**
   Measure improvement in model confidence.

4. **Store edit in memory**
   Only high-utility edits remain.

5. **Replay + new data → batch**
   Balance between stability and plasticity.

6. **Train**
   DistilBERT is updated gradually.

## 🐛 Troubleshooting

### Common Issues

1. **LLM Connection Errors**
   - Ensure Ollama server is running for LLM mode
   - Check API keys for OpenAI backend

2. **CUDA Out of Memory**
   - Reduce batch size in config
   - Set `device: cpu` if GPU memory is limited

3. **Installation Issues**
   - Use Python 3.9 or higher
   - Create a fresh virtual environment

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## 📊 Outputs
Generated under `/outputs/`:
- Training loss curve
- Accuracy curve
- JSON results
- System resource logs
- Comparison summaries (local vs llm)

## 🎯 Project Goals
- Build a resource-friendly continual learning system
- Demonstrate SEAL-style self-edit adaptive learning
- Overcome catastrophic forgetting
- Integrate local & LLM-based edits
- Achieve high retention across tasks

## 📚 References
- SEAL: Self-Edit Accumulation Learning (2025)
- Continual Learning literature
- HuggingFace Transformers
- Ollama Local LLM runtime

## 📝 Future Enhancements
- Multi-task continual evaluation (IMDB → SQuAD → ARC)
- Visualization dashboard (Streamlit)
- GPU training support
- Edit-quality analysis
- Memory pruning strategies

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
