#!/bin/bash

# A script to perform a grid search for the optimal CUDA kernel parameters
# using Nsight Compute (ncu) for precise measurements.

# --- Configuration ---
APP_NAME="./17-parallel-histogram-hybrid"
NVCC="nvcc"
NCU_PATH="/usr/local/cuda-12.5/bin/ncu" 
RESULTS_FILE="grid_search_results.csv"

# Define the parameter ranges to test
THREADS_OPTIONS="128 256 512 1024"
COARSE_OPTIONS="4 8 16 32"
MULTIPLIER_OPTIONS="2 4 8 16"

# --- Compilation ---
echo "Compiling the program..."
$NVCC 17-parallel-histogram-hybrid.cu -o $APP_NAME -O3

# --- Grid Search ---
# Create the CSV header
echo "Threads,CoarseFactor,Multiplier,ElapsedCycles,DurationMS" > $RESULTS_FILE

# Loop through all combinations
for threads in $THREADS_OPTIONS; do
  for coarse in $COARSE_OPTIONS; do
    for multiplier in $MULTIPLIER_OPTIONS; do
      
      echo "Testing: Threads=$threads, Coarse=$coarse, Multiplier=$multiplier"
      
      # Run NCU
      CMD="sudo $NCU_PATH --metrics sm__cycles_elapsed.avg,gpu__time_duration.avg $APP_NAME $threads $coarse $multiplier"
      PROFILER_OUTPUT=$(eval $CMD 2>&1)

      # Parse the profiler output to get the values
      CYCLES=$(echo "$PROFILER_OUTPUT" | grep "sm__cycles_elapsed.avg" | awk '{print $NF}' | tr -d ',')
      DURATION=$(echo "$PROFILER_OUTPUT" | grep "gpu__time_duration.avg" | awk '{print $NF}' | tr -d ',')

      # Append the parameters and parsed results to the CSV file
      echo "$threads,$coarse,$multiplier,$CYCLES,$DURATION" >> $RESULTS_FILE
      
    done
  done
done

echo "Grid search has finished, results have been saved in $RESULTS_FILE"

echo ""
echo "--- Top 5 Fastest Configurations (by Duration) ---"
(head -n 1 $RESULTS_FILE && tail -n +2 $RESULTS_FILE | sort -t, -k5 -n | head -n 5) | column -s, -t