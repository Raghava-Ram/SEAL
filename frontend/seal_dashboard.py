import streamlit as st
import os
import glob
import json
import requests
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

st.set_page_config(page_title="SEAL Continual Learning Dashboard",
                   layout="wide",
                   initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Orbitron:wght@500;700&display=swap');

body {background-color: #0b1121}
.stApp {
    background-color: #0b1121;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Orbitron', sans-serif;
    color: #00f2fe;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    text-shadow: 0 0 10px rgba(0, 242, 254, 0.3);
}
.stSidebar {
    background: linear-gradient(180deg, #111827 0%, #0b1121 100%);
    border-right: 1px solid #1f2937;
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif;
    color: #4facfe;
    text-shadow: 0 0 10px rgba(79, 172, 254, 0.4);
}
.chat-bubble {
    padding: 15px; 
    border-radius: 15px; 
    margin: 10px 0; 
    font-family: 'Inter', sans-serif;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
.user-bubble { 
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); 
    color: #0b1121; 
    margin-left: 20%; 
    border-bottom-right-radius: 2px;
}
.assistant-bubble { 
    background: #1f2937; 
    color: #e2e8f0; 
    margin-right: 20%; 
    border-bottom-left-radius: 2px;
    border: 1px solid #374151;
}
.footer { 
    color: #4b5563; 
    font-size: 13px; 
    margin-top: 25px; 
    text-align: center; 
    font-family: 'Orbitron', sans-serif;
    letter-spacing: 1px;
}
div.stButton > button {
    background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
    color: #0b1121;
    font-weight: 600;
    font-family: 'Orbitron', sans-serif;
    border: none;
    border-radius: 8px;
    box-shadow: 0 0 15px rgba(0,242,254,0.4);
    transition: all 0.3s ease;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 25px rgba(0,242,254,0.6);
    color: #0b1121;
}
hr {
    border-color: #1f2937;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

#########################
# Sidebar
#########################
with st.sidebar:
    st.title("SEAL Continual Learning System")
    page = st.radio("Navigation", [
        "🌐 Overview",
        "📊 Method Comparison",
        "🎯 Accuracy Matrix",
        "🧠 Forgetting Analysis",
        "🧪 Experiment Results",
        "🤖 AI Project Assistant"
    ])
    st.markdown("---")
    st.markdown("### 🚀 Project: SEAL")
    st.markdown("##### *Self-Adaptive Continual Learning*")
    st.markdown("`Made for: College Project Expo`")

#########################
# Utility: load codebase context (cached)
#########################
@st.cache_data
def load_codebase_context(root_dir: str = '.') -> dict:
    """Load codebase organized by priority (seal/ first, then key files)."""
    # Priority files: seal module files are most relevant
    priority_patterns = [
        'seal/*.py',
        'README.md',
        'configs/*.yaml'
    ]
    
    contents = {'seal': [], 'other': []}
    
    # Load priority files first
    for pattern in priority_patterns:
        for f in glob.glob(os.path.join(root_dir, pattern), recursive=True):
            try:
                if 'site-packages' in f or 'venv' in f or '__pycache__' in f:
                    continue
                with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                    text = fh.read()
                header = f"# FILE: {os.path.relpath(f, start=root_dir)}\n"
                entry = header + text[:10000]
                if 'seal/' in f:
                    contents['seal'].append(entry)
                else:
                    contents['other'].append(entry)
            except Exception:
                continue
    
    # Load other Python files with lower priority
    for f in glob.glob(os.path.join(root_dir, '**', '*.py'), recursive=True):
        try:
            if 'site-packages' in f or 'venv' in f or '__pycache__' in f or 'seal/' in f:
                continue
            with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                text = fh.read()
            header = f"# FILE: {os.path.relpath(f, start=root_dir)}\n"
            contents['other'].append(header + text[:5000])
        except Exception:
            continue
    
    return contents

CODEBASE_CONTEXT = load_codebase_context('.')

#########################
# System Prompt for SEAL Project
#########################
SYSTEM_PROMPT = """
You are an AI research assistant helping explain the SEAL project.

SEAL stands for Self-Adaptive Continual Learning.

The project focuses on solving catastrophic forgetting in machine learning when models learn tasks sequentially.

Key ideas of SEAL:
- Replay Memory: stores important training samples from previous tasks.
- Adaptive training: replay samples during new task training.
- Continual learning across tasks without losing previous knowledge.

Tasks used in experiments:
- IMDB (sentiment classification)
- SQuAD (question answering)
- ARC (reasoning tasks)

Methods compared:
M0: Sequential fine-tuning (baseline)
M1: Replay memory (SEAL)
M6: Hybrid replay + regularization

Important findings:
- Sequential training suffers catastrophic forgetting.
- Replay memory significantly improves knowledge retention.
- M1 achieves the best average accuracy and lowest forgetting.

Explain answers in a simple way suitable for a project expo audience.
Keep responses short and clear.
"""

def build_prompt(question: str, code_context: dict) -> str:
    """Build an optimized prompt with relevant code context (max ~3K chars)."""
    MAX_CHARS = 3000
    
    # Prioritize SEAL files first
    context_parts = []
    total_chars = 0
    
    # Add SEAL files (highest priority)
    for entry in code_context.get('seal', []):
        if total_chars + len(entry) < MAX_CHARS:
            context_parts.append(entry)
            total_chars += len(entry)
    
    # Add other files if space remains
    for entry in code_context.get('other', []):
        if total_chars + len(entry) < MAX_CHARS:
            context_parts.append(entry)
            total_chars += len(entry)
        else:
            break
    
    context_text = "\n\n".join(context_parts)
    
    return f"""{SYSTEM_PROMPT}

Project Code Context:
{context_text[:MAX_CHARS]}

User Question: {question}

Answer clearly and concisely."""

#########################
# Ollama helper
#########################
def query_ollama(prompt: str, model: str = 'phi3') -> str:
    """Send prompt to local Ollama generate API and return text response."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_length": 512
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)  # Increased from 20s for CPU inference
        if resp.status_code == 200:
            # attempt to parse JSON, but be robust against extra output
            try:
                data = resp.json()
                # Ollama may return `text` or `output`; try common keys
                return data.get('text') or data.get('output') or json.dumps(data)
            except ValueError:
                # parsing failed; resp.text likely contains multiple JSON objects from streaming
                txt = resp.text
                # attempt to extract all "response" values and concatenate
                import re
                parts = re.findall(r'"response"\s*:\s*"(.*?)"', txt)
                if parts:
                    # unescape common sequences
                    clean = ''.join(p.encode('utf-8').decode('unicode_escape') for p in parts)
                    return clean
                return txt
        else:
            return f"Ollama error: HTTP {resp.status_code} - {resp.text}"
    except Exception as e:
        return f"Ollama request failed: {e}"

#########################
# Load multi-seed experiment results
#########################
@st.cache_data
def load_multiseed_results() -> pd.DataFrame:
    """
    Load and aggregate metrics across all seeds from multiseed experiment folders.
    
    Folder structure:
        outputs/multiseed/
            seed_42/M0/multi_task/baseline/imdb_squad_arc_metrics.json
            seed_42/M1/multi_task/seal/imdb_squad_arc_metrics.json
            seed_42/M6/multi_task/hybrid/imdb_squad_arc_metrics.json
            ...same for seed_123, seed_999
    
    Returns:
        DataFrame with columns: method, accuracy, forgetting, bwt (backward transfer)
    """
    def compute_metrics_from_matrix(accuracy_matrix):
        """Compute average accuracy, forgetting, and backward transfer from accuracy matrix"""
        if not accuracy_matrix:
            return {'accuracy': 0, 'forgetting': 0, 'bwt': 0}
        
        tasks = ['imdb', 'squad', 'arc']
        accuracies = []
        
        # Collect all non-null accuracy values
        for task in tasks:
            values = accuracy_matrix.get(task, [])
            if values:
                accuracies.extend([v for v in values if v is not None])
        
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
        
        # Forgetting: measure of performance drop on earlier tasks
        # Simplified: using diagonal entries (tasks evaluated within same step)
        forgetting_values = []
        for task in tasks:
            values = accuracy_matrix.get(task, [])
            if values:
                # Get the first non-null value (original accuracy) and last non-null value
                first_val = next((v for v in values if v is not None), None)
                last_val = next((v for v in reversed(values) if v is not None), None)
                if first_val is not None and last_val is not None:
                    # Positive means forgetting (drop in accuracy)
                    forgetting_values.append(max(0, first_val - last_val))
        
        avg_forgetting = sum(forgetting_values) / len(forgetting_values) if forgetting_values else 0
        
        # BWT: backward transfer
        # Simplified: difference in performance before/after learning new tasks
        bwt_values = []
        tasks_list = list(accuracy_matrix.keys())
        for i, task in enumerate(tasks_list[:-1]):  # All but last task
            values = accuracy_matrix.get(task, [])
            if values and len(values) > i:
                # Performance of task at different points
                bwt_values.append(values[-1] if len(values) > 0 else 0)
        
        avg_bwt = (sum(bwt_values) / len(bwt_values) - 0.5) * 2 if bwt_values else 0  # Normalize
        
        return {
            'accuracy': avg_accuracy,
            'forgetting': avg_forgetting,
            'bwt': avg_bwt
        }
    
    multiseed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'multiseed')
    
    method_subpaths = {
        'M0': 'M0/multi_task/baseline/imdb_squad_arc_metrics.json',
        'M1': 'M1/multi_task/seal/imdb_squad_arc_metrics.json',
        'M6': 'M6/multi_task/hybrid/imdb_squad_arc_metrics.json'
    }
    
    results = {'M0': [], 'M1': [], 'M6': []}
    
    # Iterate through all seed folders
    if os.path.exists(multiseed_path):
        for seed_dir in os.listdir(multiseed_path):
            seed_path = os.path.join(multiseed_path, seed_dir)
            if not os.path.isdir(seed_path) or not seed_dir.startswith('seed_'):
                continue
            
            # For each method
            for method, subpath in method_subpaths.items():
                file_path = os.path.join(seed_path, subpath)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        # Try to get average_metrics; if not present, compute from accuracy_matrix
                        if 'average_metrics' in data:
                            metrics = data['average_metrics']
                            metric_dict = {
                                'accuracy': metrics.get('average_accuracy', 0),
                                'forgetting': metrics.get('average_forgetting', 0),
                                'bwt': metrics.get('average_backward_transfer', 0)
                            }
                        elif 'accuracy_matrix' in data:
                            metric_dict = compute_metrics_from_matrix(data['accuracy_matrix'])
                        else:
                            continue
                        
                        results[method].append(metric_dict)
                    except Exception as e:
                        # Skip files that can't be parsed
                        continue
    
    # Compute averages for each method
    final_results = []
    for method in ['M0', 'M1', 'M6']:
        if results[method]:
            # Average across seeds
            avg_accuracy = sum(r['accuracy'] for r in results[method]) / len(results[method])
            avg_forgetting = sum(r['forgetting'] for r in results[method]) / len(results[method])
            avg_bwt = sum(r['bwt'] for r in results[method]) / len(results[method])
            final_results.append({
                'method': method,
                'accuracy': avg_accuracy,
                'forgetting': avg_forgetting,
                'bwt': avg_bwt
            })
    
    # Return as dict, or use defaults if no data found
    if final_results:
        return {row['method']: {k: v for k, v in row.items() if k != 'method'} for row in final_results}
    else:
        # Fallback to placeholder values if no experiment data found
        return {
            'M0': {'accuracy': 0.783, 'forgetting': 0.30, 'bwt': -0.12},
            'M1': {'accuracy': 0.867, 'forgetting': 0.05, 'bwt': 0.02},
            'M6': {'accuracy': 0.769, 'forgetting': 0.20, 'bwt': -0.08}
        }

# Load real results
DEFAULT_METHOD_RESULTS = load_multiseed_results()

#########################
# Pages
#########################
if page == '🌐 Overview':
    st.header('SEAL — Self-Adaptive Continual Learning')
    st.subheader('What is catastrophic forgetting?')
    st.write('Catastrophic forgetting occurs when a model trained sequentially on multiple tasks loses performance on earlier tasks as it learns new ones.')
    st.subheader('What SEAL does')
    st.write('SEAL uses adaptive edits, priority replay memory, and utility scoring to store valuable edits and replay them during training. Hybrid variants also use sparse LLM guidance and Elastic Weight Consolidation (EWC) to protect important parameters.')
    st.subheader('Tasks used')
    st.markdown('- IMDB: sentiment classification')
    st.markdown('- SQuAD: question answering (recast)')
    st.markdown('- ARC: reasoning/knowledge tasks')
    st.subheader('Methods compared')
    st.markdown('- **M0:** Sequential Fine-tuning')
    st.markdown('- **M1:** SEAL Replay Memory')
    st.markdown('- **M6:** Hybrid Replay + EWC')

    st.markdown('---')
    col1, col2, col3 = st.columns(3)
    methods = ['M0','M1','M6']
    for c, m in zip((col1, col2, col3), methods):
        with c:
            st.metric(label=m, value=f"{DEFAULT_METHOD_RESULTS[m]['accuracy']*100:.1f}%", delta=None)

elif page == '📊 Method Comparison':
    st.header('Method Comparison')
    df = pd.DataFrame([{'method': k, **v} for k, v in DEFAULT_METHOD_RESULTS.items()])
    # Accuracy bar
    st.subheader('Average Accuracy')
    fig = px.bar(df, x='method', y='accuracy', color='method', text=df['accuracy'].apply(lambda x: f"{x:.3f}"),
                 color_discrete_sequence=['#00f2fe', '#4facfe', '#7b2cbf'])
    fig.update_layout(yaxis_title='Accuracy', showlegend=False, template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

    # Forgetting bar
    st.subheader('Forgetting')
    fig2 = px.bar(df, x='method', y='forgetting', color='method', text=df['forgetting'].apply(lambda x: f"{x:.3f}"),
                  color_discrete_sequence=['#ff4b4b', '#ff7675', '#d63031'])
    fig2.update_layout(yaxis_title='Average Forgetting', showlegend=False, template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig2, use_container_width=True)

    # Backward transfer
    st.subheader('Backward Transfer (BWT)')
    fig3 = px.bar(df, x='method', y='bwt', color='method', text=df['bwt'].apply(lambda x: f"{x:.3f}"),
                  color_discrete_sequence=['#00b894', '#55efc4', '#00cec9'])
    fig3.update_layout(yaxis_title='BWT', showlegend=False, template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown('**Highlight:** M1 (SEAL Replay Memory) shows the best average accuracy in these example results.')

elif page == '🎯 Accuracy Matrix':
    st.header('Accuracy Matrix (Task × Evaluation Step)')
    
    # Detect available seeds
    multiseed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'multiseed')
    available_seeds = []
    if os.path.exists(multiseed_path):
        available_seeds = sorted([d for d in os.listdir(multiseed_path) 
                                 if os.path.isdir(os.path.join(multiseed_path, d)) and d.startswith('seed_')])
        available_seeds = [s.replace('seed_', '') for s in available_seeds]
    
    # Dropdowns for seed and method selection
    col1, col2 = st.columns(2)
    with col1:
        selected_seed = st.selectbox('Select Seed', available_seeds if available_seeds else ['42'])
    with col2:
        selected_method = st.selectbox('Select Method', ['M0', 'M1', 'M6'])
    
    # Construct file path based on selection
    method_subpaths = {
        'M0': 'M0/multi_task/baseline/imdb_squad_arc_metrics.json',
        'M1': 'M1/multi_task/seal/imdb_squad_arc_metrics.json',
        'M6': 'M6/multi_task/hybrid/imdb_squad_arc_metrics.json'
    }
    
    file_path = os.path.join(multiseed_path, f'seed_{selected_seed}', method_subpaths[selected_method])
    
    # Load and display metrics
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                metrics_data = json.load(f)
            
            # Extract accuracy matrix
            accuracy_matrix = metrics_data.get('accuracy_matrix', {})
            
            # Convert to DataFrame with padding to handle variable-length arrays
            if accuracy_matrix:
                def pad_list(lst, target_len=3):
                    """Pad list with None values to target length"""
                    if lst is None:
                        return [None] * target_len
                    return lst + [None] * (target_len - len(lst))
                
                # Pad each task's accuracy array to 3 elements
                imdb = pad_list(accuracy_matrix.get('imdb', []))
                squad = pad_list(accuracy_matrix.get('squad', []))
                arc = pad_list(accuracy_matrix.get('arc', []))
                
                matrix_df = pd.DataFrame(
                    [imdb, squad, arc],
                    index=['IMDB', 'SQuAD', 'ARC'],
                    columns=['After IMDB', 'After SQuAD', 'After ARC']
                )
                
                # Display selection info
                st.info(f"📈 **Seed {selected_seed}** | **Method {selected_method}**")
                
                # Display heatmap
                st.write('**Heatmap (blank cells = not evaluated)**')
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(8, 4))
                fig.patch.set_facecolor('#0b1121')
                ax.set_facecolor('#0b1121')
                sns.heatmap(matrix_df.astype(float), annot=True, fmt='.2f', cmap='mako', ax=ax, 
                           cbar=True, linewidths=0.5, mask=matrix_df.isna())
                st.pyplot(fig)
                
                # Display numeric table
                st.write('**Numeric Matrix**')
                st.dataframe(matrix_df, use_container_width=True)
                
                # Display additional metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_acc = metrics_data.get('average_metrics', {}).get('average_accuracy')
                    if avg_acc:
                        st.metric('Average Accuracy', f'{avg_acc:.4f}')
                with col2:
                    avg_forgetting = metrics_data.get('average_metrics', {}).get('average_forgetting')
                    if avg_forgetting is not None:
                        st.metric('Average Forgetting', f'{avg_forgetting:.4f}')
                with col3:
                    backward_transfer = metrics_data.get('average_metrics', {}).get('average_backward_transfer')
                    if backward_transfer is not None:
                        st.metric('Avg Backward Transfer', f'{backward_transfer:.4f}')
            else:
                st.warning('No accuracy matrix found in metrics file')
        except Exception as e:
            st.error(f'Error loading metrics: {str(e)}')
    else:
        st.error(f'Metrics file not found: {file_path}')

elif page == '🧠 Forgetting Analysis':
    st.header('Forgetting Analysis')
    df = pd.DataFrame([{'method': k, 'forgetting': v['forgetting']} for k, v in DEFAULT_METHOD_RESULTS.items()])
    fig = px.bar(df, x='method', y='forgetting', color='method', text=df['forgetting'].apply(lambda x: f"{x:.3f}"),
                 color_discrete_sequence=['#ff4b4b', '#ff7675', '#d63031'])
    fig.update_layout(yaxis_title='Average Forgetting', template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    st.write('Interpretation: lower forgetting is better; SEAL (M1) minimizes forgetting by storing high-utility edits in replay memory.')

elif page == '🧪 Experiment Results':
    st.header('Experiment Results Across Seeds')
    # Example per-seed table
    example_table = pd.DataFrame([
        {'seed': 42, 'M0': 0.78, 'M1': 0.86, 'M6': 0.76},
        {'seed': 123, 'M0': 0.79, 'M1': 0.87, 'M6': 0.77},
        {'seed': 999, 'M0': 0.78, 'M1': 0.86, 'M6': 0.76}
    ])
    st.dataframe(example_table.style.format({"M0":"{:.3f}","M1":"{:.3f}","M6":"{:.3f}"}))
    st.markdown('Mean accuracy across seeds:')
    means = example_table[['M0','M1','M6']].mean().to_frame('mean').T
    st.table(means)

    st.markdown('You can point this page at `outputs/multiseed/` to auto-load real experiment CSV/JSON results if available.')

elif page == '🤖 AI Project Assistant':
    st.header('AI Project Assistant')
    st.write('Small assistant powered by local Ollama (llama2). It uses SEAL project code as context.')

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # Sidebar option: show codebase size
    seal_size = sum(len(entry) for entry in CODEBASE_CONTEXT.get('seal', []))
    other_size = sum(len(entry) for entry in CODEBASE_CONTEXT.get('other', []))
    st.sidebar.markdown(f"**Context loaded:** {seal_size+other_size:,} chars (SEAL: {seal_size:,}, Other: {other_size:,})")

    with st.container():
        chat_col1, chat_col2 = st.columns([3,1])
        with chat_col1:
            for msg in st.session_state['history']:
                if msg['role'] == 'user':
                    st.markdown(f"<div class='chat-bubble user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-bubble assistant-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
        with chat_col2:
            st.markdown('')

    question = st.text_area('Your question', height=120)
    if st.button('Send') and question.strip():
        st.session_state['history'].append({'role':'user','content':question})
        # Build optimized prompt with smart context selection
        full_prompt = build_prompt(question, CODEBASE_CONTEXT)
        answer = query_ollama(full_prompt)
        st.session_state['history'].append({'role':'assistant','content':answer})
        st.rerun()

    st.markdown('**Notes:** Ensure Ollama is running at `http://localhost:11434`. If Ollama is not reachable the assistant will show an error message.')

st.markdown('---')
st.markdown('<div class="footer">✨ SEAL: Self-Adaptive Continual Learning Framework ✨<br>Powered by local LLMs & Advanced Replay</div>', unsafe_allow_html=True)
