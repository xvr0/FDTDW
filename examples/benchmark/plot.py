import matplotlib.pyplot as plt
import sys
import numpy as np
from matplotlib.ticker import ScalarFormatter

def plot_benchmark(filename="results.csv"):
    data = {}

    try:
        with open(filename, "r") as f:
            next(f)  # Skip header
            
            for line in f:
                if not line.strip(): continue                
                parts = line.strip().split(",")
                kernel = parts[0]
                N = int(parts[1])
                time = float(parts[2])

                if kernel not in data:
                    data[kernel] = []
                
                data[kernel].append((N, time))

    except FileNotFoundError:
        print(f"File {filename} not found.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    timesteps = 2000

    for kernel, points in data.items():
        points.sort(key=lambda x: x[0])
        
        N_list = [p[0] for p in points]
        norm_time_list = [p[1] / (p[0]**3) / timesteps * 10e9 for p in points]

        if "pml" in kernel or "warp" in kernel or "split_approx" in kernel:
            current_linestyle = '-' 
        else:
            current_linestyle = ':'

        if "approx" in kernel:
            current_color = 'gold'       
        elif "fdtdx" in kernel:
            current_color = 'tab:red'
        elif "meep" in kernel:
            current_color = 'tab:blue'
        elif "warp" in kernel:
            current_color = 'darkgreen'
        else:
            current_color = 'limegreen'

        ax1.plot(
            N_list, 
            norm_time_list, 
            linestyle=current_linestyle, 
            color=current_color,
            label=kernel,
            marker='o',      
            markersize=3,    
            linewidth=1.0   
        )

    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel("$N$")
    ax1.set_ylabel(r"$t / N^3 $ [ns]")
    
    custom_ticks = [40, 60, 100, 200, 400]
    ax1.set_xticks(custom_ticks)
    ax1.set_xticklabels([str(t) for t in custom_ticks])
    ax1.xaxis.set_major_formatter(ScalarFormatter())   
    
    ax2 = ax1.twiny()     
    ax2.set_xscale('log')
    ax2.set_xlim(ax1.get_xlim())
    
    ax2.set_xticks(custom_ticks)
    
    def calculate_pml_frac(n):
        thickness = 20
        if n <= 2 * thickness:
            return 1.0 
        return 1.0 - ((n-40)**3 / n**3)

    tick_labels = [f"{calculate_pml_frac(n)*100:.0f}%" for n in custom_ticks]
    ax2.set_xticklabels(tick_labels)
    
    ax2.set_xlabel(r"$V_{pml}/V$")
    
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.4, linewidth=0.5)
    plt.tight_layout()
    
    output_file = "benchmark.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    plot_benchmark(fname)
