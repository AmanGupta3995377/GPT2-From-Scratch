import streamlit as st
import torch
import time

from inference import GPTInference

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="GPT-2 From Scratch",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    return GPTInference()

model = load_model()

DEVICE = "CUDA 🚀" if torch.cuda.is_available() else "CPU"

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]{
    font-family: 'Inter', sans-serif;
}

.main{
    background:#f5f7fb;
}

.block-container{
    max-width:1200px;
    padding-top:2rem;
    padding-bottom:2rem;
}

.hero{

    padding:40px;

    border-radius:20px;

    background:linear-gradient(135deg,#0f172a,#1e3a8a);

    color:white;

    margin-bottom:30px;

    box-shadow:0px 10px 30px rgba(0,0,0,.15);

}

.hero h1{

    font-size:52px;

    margin-bottom:5px;

}

.hero p{

    font-size:20px;

    color:#dbeafe;

}

.card{

    background:white;

    border-radius:16px;

    padding:24px;

    margin-bottom:25px;

    box-shadow:0 4px 18px rgba(0,0,0,.06);

    border:1px solid #edf2f7;

}

.metric-card{

    background:white;

    border-radius:16px;

    text-align:center;

    padding:20px;

    border:1px solid #e5e7eb;

    transition:.25s;

}

.metric-card:hover{

    transform:translateY(-4px);

    box-shadow:0 8px 22px rgba(0,0,0,.10);

}

.metric-title{

    color:#64748b;

    font-size:14px;

}

.metric-value{

    font-size:30px;

    font-weight:700;

    color:#111827;

}

.arch-box{

    background:#ffffff;

    border:1px solid #e5e7eb;

    border-radius:14px;

    padding:18px;

    text-align:center;

    font-weight:600;

    margin-bottom:10px;

}

.section-title{

    font-size:30px;

    font-weight:700;

    margin-top:10px;

    margin-bottom:20px;

}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HERO
# ==========================================================

st.markdown("""
<div class="hero">

<h1>🧠 GPT-2 From Scratch</h1>

<p>
Character-Level Decoder-Only Transformer Language Model
</p>

<p style="margin-top:15px;font-size:17px;">
Implemented completely from scratch using PyTorch.
Trained on the Tiny Shakespeare dataset and deployed with Streamlit.
</p>

</div>
""", unsafe_allow_html=True)

# ==========================================================
# PROJECT OVERVIEW
# ==========================================================

st.markdown("""
<div class="card">

<h2>📖 Project Overview</h2>

<p style="font-size:17px;line-height:1.8;">

This project implements a GPT-2 style decoder-only Transformer
completely from scratch without relying on Hugging Face model
implementations.

The objective of the project is to understand every major component
inside modern Large Language Models including:

</p>

<ul style="font-size:17px;line-height:2;">

<li>Character-level Tokenization</li>

<li>Token Embeddings</li>

<li>Positional Embeddings</li>

<li>Scaled Dot Product Attention</li>

<li>Multi-Head Self Attention</li>

<li>Feed Forward Networks</li>

<li>Residual Connections</li>

<li>Layer Normalization</li>

<li>Autoregressive Text Generation</li>

</ul>

</div>
""", unsafe_allow_html=True)

# ==========================================================
# MODEL STATISTICS
# ==========================================================

st.markdown('<div class="section-title">📊 Model Statistics</div>', unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)

cards = [
    ("Parameters","10.7M"),
    ("Layers","6"),
    ("Heads","6"),
    ("Embedding","384"),
    ("Block Size","256"),
    ("Device",DEVICE)
]

cols=[c1,c2,c3,c4,c5,c6]

for col,(title,value) in zip(cols,cards):

    with col:

        st.markdown(f"""
        <div class="metric-card">

        <div class="metric-title">{title}</div>

        <div class="metric-value">{value}</div>

        </div>
        """, unsafe_allow_html=True)

st.write("")

# ==========================================================
# ARCHITECTURE
# ==========================================================

st.markdown('<div class="section-title">🏗️ Transformer Architecture</div>', unsafe_allow_html=True)

flow = [
    "Character Tokenizer",
    "Token Embedding",
    "Positional Embedding",
    "6 × Transformer Blocks",
    "Multi-Head Attention",
    "Feed Forward",
    "Layer Normalization",
    "Language Modeling Head",
    "Autoregressive Generation"
]

for item in flow:

    st.markdown(
        f'<div class="arch-box">{item}</div>',
        unsafe_allow_html=True,
    )

    if item != flow[-1]:
        st.markdown(
            "<div style='text-align:center;font-size:24px;'>⬇️</div>",
            unsafe_allow_html=True,
        )

st.divider()

# ==========================================================
# GPT PLAYGROUND
# ==========================================================

st.markdown(
    """
<div class="section-title">
🚀 GPT Playground
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([3, 1], gap="large")

# ==========================================================
# LEFT PANEL
# ==========================================================

with left:

    st.markdown(
        """
<div class="card">
<h3>Prompt</h3>
<p style="color:#64748b;">
Enter a starting prompt and let the GPT-2 model continue it.
</p>
</div>
""",
        unsafe_allow_html=True,
    )

    prompt = st.text_area(
        "",
        value="ROMEO:",
        height=220,
        placeholder="Type your prompt here...",
        label_visibility="collapsed",
    )

# ==========================================================
# RIGHT PANEL
# ==========================================================

with right:

    st.markdown(
        """
<div class="card">

<h3>⚙ Generation Settings</h3>

</div>
""",
        unsafe_allow_html=True,
    )

    temperature = st.slider(
        "Temperature",
        0.2,
        2.0,
        1.0,
        0.1,
        help="Higher temperature produces more diverse text.",
    )

    max_tokens = st.slider(
        "Max New Tokens",
        50,
        1000,
        300,
        50,
    )

    st.write("")

    st.markdown("### Example Prompts")

    examples = [
        "ROMEO:",
        "JULIET:",
        "KING:",
        "QUEEN:",
        "LOVE:",
        "To be,",
        "Enter KING HENRY:",
        "The moon",
    ]

    example = st.selectbox(
        "",
        examples,
        label_visibility="collapsed",
    )

    if st.button(
        "📥 Load Example",
        use_container_width=True,
    ):
        st.session_state.prompt = example

# ==========================================================
# GENERATE BUTTON
# ==========================================================

st.write("")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    generate = st.button(
        "🚀 Generate Text",
        use_container_width=True,
    )

st.write("")

# ==========================================================
# GENERATION
# ==========================================================

if generate:

    start_time = time.time()

    with st.spinner("Generating Shakespeare-style text..."):

        generated_text = model.generate(
            prompt=prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

    generation_time = time.time() - start_time

    st.success("Generation Completed Successfully")

    st.markdown(
        """
<div class="section-title">
Generated Text
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<style>

.output-box{

background:white;

padding:30px;

border-radius:18px;

border:1px solid #e5e7eb;

box-shadow:0px 5px 15px rgba(0,0,0,.05);

font-size:17px;

line-height:1.8;

white-space:pre-wrap;

font-family:Georgia, serif;

}

</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
<div class="output-box">

{generated_text}

</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")

    st.download_button(
        "📥 Download Generated Text",
        generated_text,
        file_name="generated_text.txt",
        mime="text/plain",
        use_container_width=True,
    )

    words = len(generated_text.split())
    chars = len(generated_text)

    st.write("")

    st.markdown(
        """
<div class="section-title">
Generation Statistics
</div>
""",
        unsafe_allow_html=True,
    )

    a, b, c, d = st.columns(4)

    with a:

        st.metric(
            "Generation Time",
            f"{generation_time:.2f}s",
        )

    with b:

        st.metric(
            "Words",
            words,
        )

    with c:

        st.metric(
            "Characters",
            chars,
        )

    with d:

        st.metric(
            "Temperature",
            temperature,
        )

st.divider()
# ==========================================================
# MODEL INFORMATION
# ==========================================================

st.markdown(
    """
<div class="section-title">
Model Information
</div>
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([2, 1], gap="large")

with col1:

    st.markdown(
        """
<div class="card">

### About this Model

This application demonstrates a GPT-2 style decoder-only Transformer
implemented entirely from scratch using PyTorch.

The model is trained on the Tiny Shakespeare dataset and performs
character-level autoregressive text generation.

It predicts one character at a time based on the previously generated
context, producing coherent Shakespeare-style text.

</div>
""",
        unsafe_allow_html=True,
    )

with col2:

    st.markdown(
        """
<div class="card">

### Configuration

| Setting | Value |
|---------|------:|
| Architecture | GPT-2 |
| Layers | 6 |
| Attention Heads | 6 |
| Embedding Size | 384 |
| Block Size | 256 |
| Dropout | 0.2 |
| Dataset | Tiny Shakespeare |

</div>
""",
        unsafe_allow_html=True,
    )

# ==========================================================
# FEATURES
# ==========================================================

st.markdown(
    """
<div class="section-title">
Features
</div>
""",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)

with c1:

    st.markdown(
        """
<div class="card">

### 🧠 Transformer

- Decoder-only architecture
- Multi-head self attention
- Feed Forward Network
- Residual Connections

</div>
""",
        unsafe_allow_html=True,
    )

with c2:

    st.markdown(
        """
<div class="card">

### ⚡ Inference

- Checkpoint Loading
- Character Generation
- Temperature Sampling
- GPU / CPU Support

</div>
""",
        unsafe_allow_html=True,
    )

with c3:

    st.markdown(
        """
<div class="card">

### 📦 Technologies

- PyTorch
- Streamlit
- Python
- Transformer

</div>
""",
        unsafe_allow_html=True,
    )

st.divider()

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
"""
<div style="
text-align:center;
color:#94a3b8;
font-size:14px;
padding:20px;
">

GPT-2 From Scratch • PyTorch • Streamlit • Character-Level Language Model

</div>
""",
unsafe_allow_html=True,
)