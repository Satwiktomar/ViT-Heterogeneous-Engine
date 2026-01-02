import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import json

# Configuration
LOG_FILE = "logs/training_log.json"
st.set_page_config(page_title="ViT Accelerator", layout="wide")

# CSS Styling (loads from assets/style.css)
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("dashboard/assets/style.css")

# --- UI Layout ---
st.title("🚀 Heterogeneous ViT Accelerator Dashboard")

# Sidebar Controls
st.sidebar.header("Control Panel")
backend = st.sidebar.selectbox(
    "Compute Backend", 
    ["CUDA (Custom)", "OpenCL", "PyTorch (CPU/MPS)"]
)
st.sidebar.markdown(f"**Status:** Active Backend: `{backend}`")

# Main Metrics Area
col1, col2, col3, col4 = st.columns(4)
metric_throughput = col1.empty()
metric_loss = col2.empty()
metric_epoch = col3.empty()
metric_mem = col4.empty()

# Initialize metrics
metric_throughput.metric("Throughput", "0 img/s")
metric_loss.metric("Loss", "0.000")
metric_epoch.metric("Epoch", "0")
metric_mem.metric("GPU Mem", "0 MB")

st.subheader("Training Dynamics")
chart_placeholder = st.empty()

# Simulation Loop (Reads from logs in real-time)
def load_data():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                # In a real app, you'd read the last line of a JSONL file
                return json.load(f)
        except:
            return None
    return None

st.info("Waiting for training logs... (Run 'python -m vit_engine.engine.trainer' to start)")

while True:
    data = load_data()
    if data:
        # Update UI with latest numbers
        metric_throughput.metric("Throughput", f"{data.get('throughput', 0)} img/s")
        metric_loss.metric("Loss", f"{data.get('loss', 0):.4f}")
        metric_epoch.metric("Epoch", f"{data.get('epoch', 0)}")
        metric_mem.metric("GPU Mem", f"{data.get('vram', 0)} MB")
    
    time.sleep(1)