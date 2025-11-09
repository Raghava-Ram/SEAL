# Lightweight SEAL: Self-Edit Adaptive Learning

A lightweight implementation of the SEAL (Self-Edit Adaptive Learning) framework for CPU-based model adaptation, featuring offline evaluation with the IMDB dataset.

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Implementation Details](#-implementation-details)
- [Metrics and Performance](#-metrics-and-performance)
- [Troubleshooting](#-troubleshooting)
- [Future Work](#-future-work)
- [License](#-license)

## 🌟 Project Overview
- **Local Simulation**: Test editing functionality without API calls
- **OpenAI Ready**: Simple switch to use OpenAI API when needed

## 🛠️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/Raghava-Ram/SEAL.git
cd SEAL
```

### 2. Create and activate Conda environment

```bash
conda create -n seal_env python=3.10 -y
conda activate seal_env
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🧪 Testing

Run the test script to verify everything works:

```bash
python test_seal_cpu.py
```

Expected output:
```
✅ Model ran successfully on CPU
Generated Edit: Machine learning might be transforming this world.
```

## 🏗️ Project Structure

- `seal/`: Core SEAL implementation
  - `adapter.py`: Local edit simulation and mode switching
  - `openai_edit.py`: OpenAI API integration (for future use)
- `configs/`: Configuration files
  - `default.yaml`: Main configuration
- `test_seal_cpu.py`: Test script for CPU compatibility

## 🔧 Configuration

Create a `.env` file in the project root if you plan to use OpenAI API in the future:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
