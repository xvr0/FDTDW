#!/bin/bash
OUTPUT_FILE="results_fdtdw_iso.csv"
echo "Kernel,N,time" > "$OUTPUT_FILE"

KERNELS=("warp_iso") #("warp" "pml" "yee" "warp_iso")
SIZES=($(seq 40 8 128) $(seq 160 32 448))

for kernel in "${KERNELS[@]}"; do
    for N in "${SIZES[@]}"; do
        
        echo "Running Benchmark: Kernel=$kernel, N=$N"
        
        output=$(python fdtdw0.py --N "$N" --kernel "$kernel" 2>&1)
        
        runtime=$(echo "$output" | grep "simulation executed in" | awk '{print $4}' | sed 's/s//')
        
        echo "$kernel,$N,$runtime" >> "$OUTPUT_FILE"
        
    done
done

