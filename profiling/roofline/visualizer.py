import matplotlib.pyplot as plt
import numpy as np
import os

def plot_roofline(peak_flops=14.0, mem_bandwidth=0.9):
    """
    Generates a Roofline plot to check if we are Compute Bound or Memory Bound.
    Default values are approx for a laptop GPU (e.g. RTX 3050/3060).
    """
    # X-Axis: Operational Intensity (FLOPs / Byte)
    oi = np.logspace(-2, 2, 100)
    
    # Y-Axis: Performance = min(Peak, Bandwidth * OI)
    performance = np.minimum(peak_flops, mem_bandwidth * oi)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(oi, performance, 'b-', linewidth=2, label='Hardware Limit')
    
    # Add labels
    plt.xlabel('Operational Intensity (FLOPs/Byte)')
    plt.ylabel('Performance (TFLOPS)')
    plt.title('Roofline Model: ViT Kernels')
    plt.grid(True, which="both", ls="-")
    
    # Plot our Kernel Points (Theoretical)
    plt.plot(10, 10, 'ro', label='FlashAttention (Compute Bound)')
    plt.plot(0.5, 0.45, 'go', label='Standard Attn (Memory Bound)')
    
    plt.legend()
    
    output_path = 'profiling/roofline/roofline_chart.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"✅ Roofline chart saved to {output_path}")

if __name__ == "__main__":
    plot_roofline()