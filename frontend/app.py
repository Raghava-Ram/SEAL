#!/usr/bin/env python3
"""
SEAL Frontend: Streamlit-based visualization and explanation interface.

This frontend is READ-ONLY and serves to visualize results from the SEAL backend.
No training or model modification is performed here.
"""

import streamlit as st
import json
import os
from pathlib import Path
import requests
from typing import Dict, List, Optional, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import method results
from data.accuracy_matrices import METHOD_RESULTS

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "outputs" / "multi_task"
SCREENSHOTS_PATH = PROJECT_ROOT / "frontend" / "assets" / "screenshots"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama2"
MAX_CHATBOT_CONTEXT_TOKENS = 2000  # Conservative limit for llama2

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_data
def load_metrics_json(approach: str = "hybrid") -> Optional[Dict]:
    """Load accuracy metrics from JSON."""
    json_path = DATA_PATH / approach / "imdb_squad_arc_metrics.json"
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error loading metrics: {e}")
            return None
    return None


@st.cache_data
def load_task_results(approach: str = "hybrid") -> Optional[Dict]:
    """Load task results from JSON."""
    json_path = DATA_PATH / approach / "task_results.json"
    if json_path.exists():
        try:
            with open(json_path, "r") as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Error loading task results: {e}")
            return None
    return None


def check_ollama_available() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def call_ollama(prompt: str, system_prompt: str = "") -> Tuple[bool, str]:
    """
    Call Ollama API to generate a response.
    
    Returns:
        Tuple of (success: bool, response: str)
    """
    try:
        url = f"{OLLAMA_BASE_URL}/api/generate"
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "system": system_prompt,
        }
        # Increased timeout to 120 seconds for slower systems
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            data = response.json()
            return True, data.get("response", "No response generated.")
        else:
            return False, f"Ollama error: {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Ollama is responding slowly. This can happen on first request or with slower hardware. Please try again or increase timeout if needed."
    except Exception as e:
        return False, f"Error calling Ollama: {str(e)}"


def compute_forgetting(accuracy_matrix: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Compute forgetting for each task.
    
    Forgetting = max accuracy (before task completed) - final accuracy
    """
    forgetting = {}
    for task, accs in accuracy_matrix.items():
        valid_accs = [acc for acc in accs if acc is not None]
        if len(valid_accs) <= 1:
            forgetting[task] = 0.0
        else:
            max_acc = max(valid_accs[:-1])
            final_acc = valid_accs[-1]
            forgetting[task] = max(0.0, max_acc - final_acc)
    return forgetting


def format_matrix_as_table(accuracy_matrix: Dict[str, List[float]]) -> pd.DataFrame:
    """Format accuracy matrix as a pandas DataFrame for nice display."""
    # Find max sequence length
    max_len = max(len(accs) for accs in accuracy_matrix.values())
    
    # Create columns: Task | Step 1 | Step 2 | ... | Step N
    columns = ["Task"] + [f"Step {i+1}" for i in range(max_len)]
    rows = []
    
    for task, accs in accuracy_matrix.items():
        row = [task]
        for acc in accs:
            # Pad with empty or format
            row.append(f"{acc:.2%}" if acc is not None else "—")
        # Pad with empty strings if needed
        while len(row) < len(columns):
            row.append("—")
        rows.append(row)
    
    return pd.DataFrame(rows, columns=columns)


def matrix_list_to_dataframe(matrix: List[List[Optional[float]]]) -> pd.DataFrame:
    """
    Convert a 3x3 matrix list to a formatted DataFrame for heatmap display.
    
    Matrix format:
    - Row 0: IMDB accuracies (after task 1, 2, 3)
    - Row 1: SQuAD accuracies (after task 2, 3)
    - Row 2: ARC accuracies (after task 3)
    """
    tasks = ["IMDB", "SQuAD", "ARC"]
    steps = ["After Task 1", "After Task 2", "After Task 3"]
    
    # Build dataframe with proper indexing
    data = []
    for i, task in enumerate(tasks):
        row = {}
        for j in range(3):
            if j >= i:  # Upper triangular (only computed after task i)
                row[steps[j]] = matrix[i][j]
            else:
                row[steps[j]] = None
        data.append(row)
    
    df = pd.DataFrame(data, index=tasks)
    return df


def render_accuracy_heatmap(matrix: List[List[Optional[float]]], title: str = "Accuracy Matrix"):
    """Render accuracy matrix as a heatmap."""
    df = matrix_list_to_dataframe(matrix)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Replace None with NaN for visualization
    df_numeric = df.copy()
    df_numeric = df_numeric.astype(float)
    
    # Create heatmap
    sns.heatmap(
        df_numeric,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Accuracy"},
        ax=ax,
        linewidths=0.5,
        linecolor="gray"
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Task", fontsize=12)
    ax.set_xlabel("Training Step", fontsize=12)
    
    return fig


def get_method_names_ordered() -> List[str]:
    """Get list of method names in order (M0–M6)."""
    return sorted(METHOD_RESULTS.keys())


def compute_forgetting_for_matrix(matrix: List[List[Optional[float]]]) -> Dict[str, float]:
    """
    Compute forgetting score for each task from a 3x3 matrix.
    
    Args:
        matrix: 3x3 upper triangular matrix
        
    Returns:
        Dict with tasks as keys and forgetting (as %) as values
    """
    tasks = ["IMDB", "SQuAD", "ARC"]
    forgetting = {}
    
    for i, task in enumerate(tasks):
        task_accs = [acc for acc in matrix[i] if acc is not None]
        if len(task_accs) > 1:
            max_acc = max(task_accs[:-1])
            final_acc = task_accs[-1]
            forget_score = max(0.0, (max_acc - final_acc) * 100)  # Convert to percentage
            forgetting[task] = forget_score
        else:
            forgetting[task] = 0.0
    
    return forgetting


def render_dual_accuracy_plots(matrix: List[List[Optional[float]]]) -> plt.Figure:
    """
    Render accuracy trends and forgetting analysis side by side.
    Similar to the user's provided visualization.
    
    Args:
        matrix: 3x3 upper triangular matrix
        
    Returns:
        matplotlib figure with two subplots
    """
    tasks = ["imdb", "squad", "arc"]
    
    # Extract accuracy data
    accuracies = {
        "imdb": [],
        "squad": [],
        "arc": []
    }
    
    # Matrix structure:
    # Row 0 (IMDB): [Task1, Task2, Task3]
    # Row 1 (SQuAD): [None, Task2, Task3]
    # Row 2 (ARC): [None, None, Task3]
    
    # Collect accuracies at each step
    for step in range(3):
        for task_idx in range(3):
            if step >= task_idx:  # Upper triangular
                acc = matrix[task_idx][step]
                if acc is not None:
                    accuracies[tasks[task_idx]].append(acc)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Task Accuracy Over Time
    for task in tasks:
        if accuracies[task]:
            ax1.plot(range(len(accuracies[task])), accuracies[task], 
                    marker='o', label=task, linewidth=2, markersize=8)
    
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Task Accuracy Over Time", fontsize=13, fontweight="bold")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Forgetting per Task
    forgetting_data = []
    task_names = []
    
    for i, task in enumerate(tasks):
        if accuracies[task] and len(accuracies[task]) > 1:
            max_acc = max(accuracies[task][:-1])
            final_acc = accuracies[task][-1]
            forgetting = max(0.0, max_acc - final_acc) * 100  # Convert to percentage
            forgetting_data.append(forgetting)
            task_names.append(task)
    
    if forgetting_data:
        colors = ["#FF7F50" if f > 0 else "#90EE90" for f in forgetting_data]
        ax2.bar(task_names, forgetting_data, color=colors, edgecolor="black", linewidth=1.5)
        ax2.set_ylabel("Forgetting (%)", fontsize=12)
        ax2.set_title("Forgetting per Task", fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    return fig


# ============================================================================
# PAGE: OVERVIEW
# ============================================================================

def page_overview():
    """Screen 1: Project Overview"""
    st.title("🚀 SEAL: Continual Learning without Catastrophic Forgetting")
    
    st.markdown("""
    ### What is Catastrophic Forgetting?
    
    When a neural network is trained sequentially on multiple tasks, it often **forgets** 
    previously learned knowledge when learning new tasks. This phenomenon is called 
    **catastrophic forgetting** (or catastrophic interference).
    
    **Example**: A model trained first on IMDB (sentiment analysis), then SQuAD (question answering),
    then ARC (common sense reasoning) will progressively lose its ability to perform IMDB.
    """)
    
    # Simple text-based visualization
    st.markdown("""
    #### Sequential Learning Challenge
    
    ```
    Time: t1              t2              t3
          ↓               ↓               ↓
    
    Task: IMDB        SQuAD           ARC
    Accuracy: 95%     57% (↓ IMDB)    0.8% (↓ both!)
              ▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀▀▀▀
    ```
    """)
    
    st.markdown("""
    ### The SEAL Solution
    
    **SEAL** (Self-Edit Adaptive Learning) addresses catastrophic forgetting through:
    
    1. **Replay Memory**: Store high-utility past examples and replay them during training
    2. **Elastic Weight Consolidation (EWC)**: Protect important parameters with Fisher information
    3. **Task-Specific Classifiers**: Learn independent heads for each task
    4. **Adaptive Edits**: Generate or modify training examples to improve model robustness
    
    #### Method Variants
    
    - **Baseline**: No protection mechanism → severe forgetting
    - **SEAL with Replay**: Replays past examples (failure case) → demonstrates insufficiency of replay alone
    - **Hybrid SEAL**: Combines replay + task-specific heads → better but still imperfect
    - **Hybrid + Task-weighted Replay**: Alternative approach without parameter protection (failure case)
    - **Hybrid + EWC**: Adds parameter-level protection → **best stability**
    
    ### Key Insight
    
    > **Parameter-level protection (via EWC) is essential** for maintaining previous task performance 
    > while learning new tasks. Replay-only approaches are fundamentally insufficient without it.
    
    *Note: Methods without EWC are intentionally included to motivate the necessity of parameter-level protection.*
    """)
    
    st.info(
        "💡 This frontend visualizes pre-computed results from the SEAL backend. "
        "No training or model modification occurs here."
    )


# ============================================================================
# PAGE: ACCURACY MATRIX
# ============================================================================

def page_accuracy_matrix():
    """Screen 2: Accuracy Matrix Viewer - Multi-Method Comparison"""
    st.title("📊 Accuracy Matrix Viewer")
    
    st.markdown("""
    Compare continual learning strategies through their accuracy matrices.
    - **Diagonal**: Performance on the task immediately after learning it
    - **Off-diagonal**: Performance after learning subsequent tasks (forgetting visible)
    """)
    
    # Create two tabs: Pre-computed Methods vs Live Backend
    tab1, tab2 = st.tabs(["📈 Pre-computed Methods (M0–M6)", "🔄 Live Backend Results"])
    
    # TAB 1: Pre-computed Methods
    with tab1:
        st.subheader("Continual Learning Method Progression")
        
        # Get ordered method list
        method_names = get_method_names_ordered()
        
        if not method_names:
            st.warning("No methods found in accuracy_matrices.py")
            return
        
        # Method selector
        selected_method = st.selectbox(
            "Select a method to view:",
            options=method_names,
            format_func=lambda x: x,  # Display as-is (M0: Baseline, etc.)
            key="method_selector"
        )
        
        if selected_method in METHOD_RESULTS:
            method_data = METHOD_RESULTS[selected_method]
            matrix = method_data["matrix"]
            screenshot_name = method_data.get("source") or method_data.get("screenshot")  # Support both field names
            description = method_data["description"]
            
            # Display method description
            st.markdown(f"**{selected_method}**: {description}")
            
            # Display dual accuracy plots (like the user's visualization)
            st.subheader("Accuracy Analysis")
            fig_dual = render_dual_accuracy_plots(matrix)
            st.pyplot(fig_dual)
            
            # Create two columns: Heatmap + Screenshot
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Accuracy Heatmap")
                fig = render_accuracy_heatmap(matrix, title=selected_method)
                st.pyplot(fig)
                
                # Display numeric table below heatmap
                st.subheader("Numeric Values")
                df = matrix_list_to_dataframe(matrix)
                st.dataframe(df)
            
            with col2:
                st.subheader("Experimental Result")
                screenshot_path = SCREENSHOTS_PATH / screenshot_name
                
                if screenshot_path.exists():
                    st.image(str(screenshot_path))
                else:
                    st.info(
                        f"📁 Screenshot not found: `{screenshot_name}`\n\n"
                        f"Place at: `frontend/assets/screenshots/{screenshot_name}`"
                    )
            
            # Display forgetting analysis
            st.subheader("Forgetting Analysis")
            tasks = ["IMDB", "SQuAD", "ARC"]
            forgetting_vals = []
            
            for i, task in enumerate(tasks):
                task_accs = [acc for acc in matrix[i] if acc is not None]
                if len(task_accs) > 1:
                    max_acc = max(task_accs[:-1])
                    final_acc = task_accs[-1]
                    forgetting = max(0.0, max_acc - final_acc)
                else:
                    forgetting = 0.0
                forgetting_vals.append({"Task": task, "Forgetting (%)": f"{forgetting*100:.2f}%"})
            
            forgetting_df = pd.DataFrame(forgetting_vals)
            st.dataframe(forgetting_df)
            
            # Method comparison reference
            st.markdown("---")
            st.markdown("""
            ### Method Progression Reference
            - **M0**: Baseline (no protection)
            - **M1**: SEAL with Replay only
            - **M2**: Hybrid (LLM + Replay)
            - **M3**: Hybrid + Sparse LLM
            - **M4**: Hybrid + Freezing
            - **M5**: Hybrid + Task-Weighted Replay
            - **M6**: FINAL – Hybrid + EWC (best results)
            """)
    
    # TAB 2: Live Backend Results
    with tab2:
        st.subheader("Live Backend Results (JSON)")
        
        # Try to load metrics from backend
        metrics = load_metrics_json("hybrid")
        
        if metrics and "accuracy_matrix" in metrics:
            accuracy_matrix = metrics["accuracy_matrix"]
            
            st.info("✅ Loaded from: `outputs/multi_task/hybrid/imdb_squad_arc_metrics.json`")
            
            # Display as table
            df = format_matrix_as_table(accuracy_matrix)
            st.dataframe(df)
            
            # Compute forgetting
            forgetting = compute_forgetting(accuracy_matrix)
            st.subheader("Forgetting Analysis")
            forgetting_df = pd.DataFrame({
                "Task": list(forgetting.keys()),
                "Forgetting (%)": [f"{v*100:.2f}%" for v in forgetting.values()]
            })
            st.dataframe(forgetting_df)
            
            # Visualization
            st.subheader("Visualization")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 1: Accuracy trends
            for task, accs in accuracy_matrix.items():
                ax1.plot(range(len(accs)), accs, marker='o', label=task)
            ax1.set_xlabel("Training Step")
            ax1.set_ylabel("Accuracy")
            ax1.set_title("Task Accuracy Over Time")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Forgetting bars
            tasks = list(forgetting.keys())
            forget_vals = [forgetting[t]*100 for t in tasks]
            ax2.bar(tasks, forget_vals, color='coral')
            ax2.set_ylabel("Forgetting (%)")
            ax2.set_title("Forgetting per Task")
            ax2.grid(True, alpha=0.3, axis='y')
            
            st.pyplot(fig)
        else:
            st.warning(
                "⚠️ Live backend metrics not found at `outputs/multi_task/hybrid/imdb_squad_arc_metrics.json`. "
                "Please ensure the backend has been run."
            )
            st.info("Use the **Pre-computed Methods** tab to explore M0–M6 results instead.")



# ============================================================================
# PAGE: METHOD COMPARISON (Screenshots)
# ============================================================================

def page_method_comparison():
    """Screen 3: Method Comparison with Screenshots"""
    st.title("🎯 Method Comparison: From Baseline to Hybrid SEAL + EWC")
    
    st.markdown("""
    Below is a progression showing how different techniques improve continual learning performance.
    """)
    
    # Define screenshots
    screenshots = [
        {
            "name": "baseline.png",
            "title": "1. Baseline – Severe Catastrophic Forgetting",
            "caption": "Without any protection mechanism, the model rapidly forgets previous tasks.",
        },
        {
            "name": "seal_replay.png",
            "title": "2. SEAL with Replay – Partial Recovery",
            "caption": "Replaying past examples helps, but forgetting still occurs.",
        },
        {
            "name": "hybrid_llm_replay.png",
            "title": "3. Hybrid SEAL with LLM-Generated Edits – Some Improvement",
            "caption": "Combining replay with adaptive edits provides moderate protection.",
        },
        {
            "name": "hybrid_freezing.png",
            "title": "4. Hybrid + Task-Specific Freezing – Better Stability",
            "caption": "Freezing encoder parameters on old tasks improves retention.",
        },
        {
            "name": "hybrid_task_weighted_replay.png",
            "title": "5. Task-Aware Replay (Run 1) – Variable Results",
            "caption": "Weighted sampling by task shows promise but remains unstable.",
        },
        {
            "name": "hybrid_task_weighted_replay_v2.png",
            "title": "6. Task-Aware Replay (Run 2) – Consistent Pattern",
            "caption": "Repeated runs show similar trends: some tasks more robust than others.",
        },
        {
            "name": "hybrid_ewc_final.png",
            "title": "7. FINAL: Hybrid SEAL + EWC – Best Retention & Stability",
            "caption": "Elastic Weight Consolidation (EWC) provides the optimal stability-plasticity tradeoff.",
        },
    ]
    
    # Create tabs for each screenshot
    tabs = st.tabs([s["title"].split(". ")[1] for s in screenshots])
    
    for i, (tab, screenshot_info) in enumerate(zip(tabs, screenshots)):
        with tab:
            screenshot_path = PROJECT_ROOT / "frontend" / "assets" / "screenshots" / screenshot_info["name"]
            
            st.markdown(f"### {screenshot_info['title']}")
            st.markdown(f"**{screenshot_info['caption']}**")
            
            if screenshot_path.exists():
                st.image(str(screenshot_path))
            else:
                st.info(
                    f"📁 Screenshot not found: `{screenshot_info['name']}`\n\n"
                    f"Place your image files in: `frontend/assets/screenshots/`"
                )
    
    st.markdown("---")
    
    st.success("""
    ### 🏆 Key Conclusion
    
    **EWC provides the best stability–plasticity tradeoff.**
    
    - **Stability**: Protected parameters maintain prior task knowledge
    - **Plasticity**: Learning on new tasks remains effective
    - **Mechanism**: Fisher information matrix guides which parameters to protect
    """)


# ============================================================================
# PAGE: FORGETTING ANALYSIS
# ============================================================================

def page_forgetting_analysis():
    """Screen 4: Forgetting Analysis"""
    st.title("📉 Forgetting Analysis: Understanding Catastrophic Forgetting")
    
    st.markdown("""
    ### What is Forgetting?
    
    Forgetting quantifies the loss in performance on a task after learning new tasks.
    
    #### Mathematical Definition
    
    For each task **i**:
    
    ```
    Forgetting_i = max(accuracy_i during training) - final_accuracy_i
    ```
    
    A forgetting value of 0% means perfect retention; higher values indicate more forgetting.
    """)
    
    # =========================================================================
    # COMPREHENSIVE COMPARISON TABLE: ALL METHODS
    # =========================================================================
    st.subheader("📊 Forgetting Scores Across All Methods (M0–M6)")
    st.markdown("Compare how each method handles forgetting for each task:")
    
    # Build comprehensive table
    all_methods_forgetting = []
    method_names = get_method_names_ordered()
    
    for method_name in method_names:
        if method_name in METHOD_RESULTS:
            method_data = METHOD_RESULTS[method_name]
            matrix = method_data["matrix"]
            forgetting_scores = compute_forgetting_for_matrix(matrix)
            
            row = {"Method": method_name}
            row.update({f"{task} Forgetting (%)": f"{score:.2f}%" for task, score in forgetting_scores.items()})
            
            # Calculate average forgetting
            avg_forgetting = sum(forgetting_scores.values()) / len(forgetting_scores)
            row["Avg Forgetting (%)"] = f"{avg_forgetting:.2f}%"
            
            all_methods_forgetting.append(row)
    
    # Create and display comparison dataframe
    comparison_df = pd.DataFrame(all_methods_forgetting)
    
    # Display with better formatting
    st.dataframe(comparison_df, use_container_width=False)
    
    # Add interpretation guide
    st.markdown("""
    #### How to Read the Table
    
    - **Method**: Continual learning approach (M0–M6)
    - **IMDB Forgetting (%)**: % loss on IMDB after learning SQuAD and ARC
    - **SQuAD Forgetting (%)**: % loss on SQuAD after learning ARC
    - **ARC Forgetting (%)**: % loss on ARC (typically 0% since it's the last task)
    - **Avg Forgetting (%)**: Average forgetting across all tasks
    
    **Lower is better!** Methods with lower forgetting scores are more effective at retaining previous knowledge.
    """)
    
    # =========================================================================
    # SINGLE METHOD DEEP DIVE (Optional)
    # =========================================================================
    
    st.markdown("---")
    st.subheader("🔍 Deep Dive: Single Method Analysis")
    
    # Load metrics or use placeholder
    metrics = load_metrics_json("hybrid")
    
    if metrics and "accuracy_matrix" in metrics:
        accuracy_matrix = metrics["accuracy_matrix"]
    else:
        accuracy_matrix = {
            "imdb": [0.95, 0.69, 0.82],
            "squad": [0.57, 0.43],
            "arc": [0.97]
        }
        st.info("Using placeholder data for demonstration.")
    
    # Compute forgetting
    forgetting = compute_forgetting(accuracy_matrix)
    
    st.subheader("Forgetting Summary (Hybrid SEAL + EWC)")
    
    forgetting_data = []
    for task, accs in accuracy_matrix.items():
        max_acc = max([a for a in accs if a is not None][:-1]) if len([a for a in accs if a is not None]) > 1 else 0
        final_acc = [a for a in accs if a is not None][-1] if accs else 0
        forget = forgetting[task]
        forgetting_data.append({
            "Task": task.upper(),
            "Max Accuracy": f"{max_acc:.2%}",
            "Final Accuracy": f"{final_acc:.2%}",
            "Forgetting": f"{forget:.2%}",
        })
    
    df = pd.DataFrame(forgetting_data)
    st.dataframe(df)
    
    st.markdown("""
    ### Key Insights
    
    **1. Replay alone is insufficient**
    - Simply replaying past examples reduces forgetting but doesn't eliminate it
    - New task learning can still override old knowledge
    
    **2. Parameter-level protection is required**
    - EWC protects important parameters by constraining weight changes
    - This prevents overwriting critical knowledge for previous tasks
    
    **3. Task-specific heads help**
    - Separate classification heads per task reduce confusion
    - Reduces competition between different task objectives
    
    **4. Combined approach is optimal**
    - Hybrid SEAL + EWC + task-specific heads achieves best results
    - Forgetting reduced while maintaining learning capacity for new tasks
    """)
    
    st.warning("""
    ### ⚠️ The Stability-Plasticity Dilemma
    
    Too much protection → model cannot learn new tasks effectively (low plasticity)
    Too little protection → model forgets previous tasks (poor stability)
    
    **EWC elegantly balances this** by weighing parameter importance via Fisher information.
    """)


# ============================================================================
# PAGE: CHATBOT
# ============================================================================

def page_chatbot():
    """Screen 5: Conversational Chatbot"""
    st.title("💬 SEAL Chatbot: Ask Questions About Continual Learning")
    
    st.markdown("""
    This chatbot uses **llama2** (via Ollama) to answer questions about SEAL, 
    continual learning, and the experimental results.
    """)
    
    # Check Ollama availability
    if not check_ollama_available():
        st.warning(
            "Chatbot unavailable in cloud deployment. "
            "Run locally with Ollama to enable conversational explanations."
        )
        return
    
    st.success("✅ Ollama is running and ready!")
    
    # Build system context (optimized for response speed)
    system_context = """You are an expert on SEAL (Self-Edit Adaptive Learning), a continual learning framework.

Key facts:
1. Catastrophic forgetting: Models lose performance on old tasks when learning new tasks
2. SEAL components: Replay memory + EWC + task-specific heads
3. EWC protects parameters via Fisher information
4. Setup: IMDB → SQuAD → ARC sequential tasks
5. Forgetting = max_accuracy - final_accuracy

Answer clearly and concisely. Reference SEAL's approach."""

    # Load context data (minimal for speed)
    metrics = load_metrics_json("hybrid")
    
    context_data = ""
    if metrics and "accuracy_matrix" in metrics:
        context_data = "\nResults (Hybrid SEAL + EWC):"
        for task, accs in metrics["accuracy_matrix"].items():
            context_data += f"\n- {task.upper()}: {accs}"
    
    # Chat history management
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    st.subheader("Conversation")
    
    chat_container = st.container()
    
    with chat_container:
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.chat_message("user").write(message)
            else:
                st.chat_message("assistant").write(message)
    
    # Input area
    st.divider()
    
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask a question about SEAL or continual learning:",
            placeholder="e.g., 'How does EWC prevent catastrophic forgetting?' or 'Why is replay memory important?'",
            key="chat_input"
        )
        submitted = st.form_submit_button("Send")
    
    if submitted and user_input:
        # Add user message to history
        st.session_state.chat_history.append(("user", user_input))
        
        # Prepare concise prompt
        full_prompt = f"""Context:{context_data}

Question: {user_input}

Answer concisely about SEAL or continual learning."""
        
        # Get response from Ollama
        with st.spinner("🤔 Generating response..."):
            success, response = call_ollama(full_prompt, system_context)
        
        if success:
            st.session_state.chat_history.append(("assistant", response))
            st.rerun()
        else:
            st.error(f"Failed to get response: {response}")
    
    # Clear history button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main Streamlit app with sidebar navigation."""
    
    st.set_page_config(
        page_title="SEAL: Continual Learning Frontend",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar navigation
    st.sidebar.title("🧭 SEAL Frontend")
    
    page = st.sidebar.radio(
        "Select Screen:",
        options=[
            "📖 Overview",
            "📊 Accuracy Matrix",
            "🎯 Method Comparison",
            "📉 Forgetting Analysis",
            "💬 Chatbot"
        ],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About SEAL
    
    **SEAL** is a continual learning framework that prevents catastrophic forgetting 
    through adaptive edits, replay memory, and elastic weight consolidation.
    
    This frontend is **read-only** and visualizes pre-computed results from the backend.
    
    ---
    
    📁 **Data Source**: `outputs/multi_task/hybrid/`
    
    🤖 **Chatbot**: llama2 via Ollama
    """)
    
    # Route to appropriate page
    if "Overview" in page:
        page_overview()
    elif "Accuracy Matrix" in page:
        page_accuracy_matrix()
    elif "Method Comparison" in page:
        page_method_comparison()
    elif "Forgetting Analysis" in page:
        page_forgetting_analysis()
    elif "Chatbot" in page:
        page_chatbot()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
    SEAL Frontend v1.0 | Read-only visualization layer | Backend: Python | Frontend: Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
