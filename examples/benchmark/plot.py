import matplotlib.pyplot as plt
import sys
import numpy as np
from matplotlib.ticker import ScalarFormatter

def plot_benchmark(filename="results.csv"):
    data = {}

    try:
        with open(filename, "r") as f:
            header = next(f)  # Skip header
            
            for line in f:
                if not line.strip(): continue                
                parts = line.strip().split(",")
                kernel = parts[0].strip()
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

    def get_vectors(k_name):
        points = data[k_name]
        points.sort(key=lambda x: x[0])
        Ns = np.array([p[0] for p in points])
        Ts = np.array([p[1] / (p[0]**3) / timesteps * 10e9 for p in points])
        return Ns, Ts

    c_yee = 'limegreen'     
    c_pml = 'darkgreen'     
    c_warp = 'forestgreen'  
    c_split = 'gold'        
    
    plot_defs = [
        ('FDTDW(anisotrop)', 'FDTDW(isotrop)', c_warp, 'FDTDW(anisotrop)', '-'),
    ]

    plotted_kernels = set()

    for base_key, iso_key, color, label, style in plot_defs:
        if base_key in data:
            n_base, t_base = get_vectors(base_key)
            plotted_kernels.add(base_key)
            
            final_style = style
            if base_key == 'pml': final_style = '-'
            
            if iso_key in data:
                n_iso, t_iso = get_vectors(iso_key)
                plotted_kernels.add(iso_key)
                
                base_dict = dict(zip(n_base, t_base))
                iso_dict = dict(zip(n_iso, t_iso))
                common_Ns = sorted(list(set(base_dict.keys()) & set(iso_dict.keys())))
                
                if common_Ns:
                    common_Ns = np.array(common_Ns)
                    t_base_common = np.array([base_dict[n] for n in common_Ns])
                    t_iso_common = np.array([iso_dict[n] for n in common_Ns])
                    
                    ax1.fill_between(common_Ns, t_base_common, t_iso_common, 
                                     color=color, alpha=0.7, linewidth=0, zorder=1)
                
                ax1.plot(n_iso, t_iso, linestyle="--", color=color, 
                         label='FDTDW (isotrop)',   
                         marker='o',           
                         markersize=3,         
                         linewidth=1.0, alpha=0.8, zorder=2)

            ax1.plot(n_base, t_base, linestyle=final_style, color=color, 
                     label=label, marker='o', markersize=3, linewidth=1.5, zorder=3)

    for kernel in data.keys():
        if kernel in plotted_kernels: continue
        # if "isotrop" in kernel: continue 
        # if "split_approx_real" in kernel: continue
        
        n_k, t_k = get_vectors(kernel)
        
        ls = '-'
        mk = 'o'
        c = 'grey'
        zo = 4 
        
        if "FDTDX" in kernel: 
            c = 'tab:red'       
            ls = ':' if "periodic" in kernel else '-'
            
        elif "MEEP" in kernel: 
            c = 'tab:blue'      
            ls = ':' if "no PML" in kernel else '-'
            
        elif "approx" in kernel:
            c = c_split         
            ls = '-'
        elif "warp" in kernel:
            c = c_warp          
            ls = '-'
        elif "no PML" in kernel:
            c = c_yee         
            ls = ':'       
        elif "all PML" in kernel : 
             c = c_pml
             ls = '-'
             
        ax1.plot(n_k, t_k, linestyle=ls, color=c, label=kernel, 
                 marker=mk, markersize=3, zorder=zo)

    # --- Formatting ---
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlabel(r"$N_x=N_y=N_z$")
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
        if n <= 2 * thickness: return 1.0
        dim_inner = n - 2 * thickness
        vol_inner = dim_inner**3
        vol_total = n**3
        return 1.0 - (vol_inner / vol_total)

    ax2.set_xticklabels([f"{calculate_pml_frac(n)*100:.0f}%" for n in custom_ticks])
    ax2.set_xlabel(r"$V_{pml}/V$")
    
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    # ax1.legend(by_label.values(), by_label.keys(), loc='center right', fontsize='small')
    
    ax1.grid(True, which="both", ls="-", alpha=0.4, linewidth=0.5)
    plt.tight_layout()
    
    output_file = "benchmark_.png"
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
    plot_benchmark(fname)
