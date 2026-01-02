import streamlit as st

def render_gpu_stats(allocated, reserved, max_mem):
    """Custom widget to display GPU memory bars"""
    st.markdown("### GPU Memory Usage")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Allocated", f"{allocated:.2f} MB")
    col2.metric("Reserved", f"{reserved:.2f} MB")
    col3.metric("Peak", f"{max_mem:.2f} MB")
    
    # Simple progress bar for memory pressure (assuming 16GB VRAM roughly)
    pressure = min(allocated / 16000, 1.0)
    st.progress(pressure, text="VRAM Pressure")

def render_backend_selector():
    return st.sidebar.radio(
        "Compute Backend",
        ("PYTORCH", "CUDA_CUSTOM", "OPENCL"),
        index=0,
        help="Select which engine to use for inference."
    )