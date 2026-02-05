import pandas as pd
import numpy as np

def approx(input_file="results.csv", output_file="approx.csv"):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        return

    df = df[df['N'] > 100]

    df_pivot = df.pivot(index='N', columns='Kernel', values='time')
    

    N = df_pivot.index.to_numpy()
    T_yee = df_pivot['yee'].to_numpy()
    T_pml = df_pivot['pml'].to_numpy()

    FLOP_YEE = 36.0
    FLOP_PML = 96.0
    RATIO = FLOP_PML / FLOP_YEE 

    # T_yee = M_full + C_yee
    # T_pml = M_full + (Ratio * C_yee)
    
    C_yee_est = (T_pml - T_yee) / (RATIO - 1.0)
    
    M_full_est = T_yee - C_yee_est
    M_full_est = np.maximum(0, M_full_est) 

    T_yee_approx = (M_full_est / 3.0) + C_yee_est

    PML_THICKNESS = 20
    dim_core = np.maximum(0, N - 2 * PML_THICKNESS)
    vol_core = dim_core ** 3
    vol_total = N ** 3
    
    frac_core = vol_core / vol_total
    frac_pml = 1.0 - frac_core

    T_split_approx = (frac_core * T_yee_approx) + (frac_pml * T_pml)

    
    df_yee = pd.DataFrame({
        'Kernel': ['yee_approx'] * len(N),
        'N': N,
        'time': T_yee_approx
    })
    
    df_split = pd.DataFrame({
        'Kernel': ['split_approx'] * len(N),
        'N': N,
        'time': T_split_approx
    })
    
    final_df = pd.concat([df_yee, df_split], ignore_index=True)
    
    final_df = final_df.sort_values(by=['Kernel', 'N'])

    final_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    approx()
