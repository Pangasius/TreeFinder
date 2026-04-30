#!/bin/bash

# Usage: ./slurm.sbatch [dataset]
# If no dataset is given, submits jobs for ALL datasets.
# Valid datasets: hotpot_qa, loogle_short, loogle_long, longbench

ALL_DATASETS=("hotpot_qa" "loogle_short" "loogle_long" "longbench")

# If no argument, submit one job per dataset
if [ -z "$1" ]; then
    echo "No dataset specified — submitting jobs for all datasets."
    for DS in "${ALL_DATASETS[@]}"; do
        bash "$0" "$DS"
    done
    exit 0
fi

DATASET=$1

# Set resources based on dataset
case $DATASET in
    "hotpot_qa")
        GPUS=1
        TIME="1-00:00:00"
        ABLATIONS=50
        ;;
    "loogle_short")
        GPUS=2
        TIME="2-00:00:00"
        ABLATIONS=100
        ;;
    "loogle_long")
        GPUS=2
        TIME="2-00:00:00"
        ABLATIONS=100
        ;;
    "longbench")
        GPUS=2
        TIME="1-00:00:00"
        ABLATIONS=100
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Valid options: hotpot_qa, loogle_short, loogle_long, longbench"
        exit 1
        ;;
esac

echo "Submitting job for dataset: $DATASET"
echo "Configuration: $GPUS GPUs, $TIME time limit"

# Calculate resources based on GPU count
CPUS=$((7 * GPUS))
MEM=$((50 * GPUS))

# Submit the job with the appropriate parameters
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=TreeFinder_${DATASET}_%j
#SBATCH --gpus=${GPUS}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}GB
#SBATCH -o log_${DATASET}_%j.out
#SBATCH -e log_${DATASET}_%j.err
#SBATCH --account=ACCOUNT_NAME
#SBATCH --partition=gpu
#SBATCH --export=ALL

# Print configuration
echo "Running with dataset: $DATASET"
echo "GPUs: $GPUS, Time: $TIME"

# Activate the conda environment
cd ~
source ~/.bashrc
conda activate CONDA_ENV_NAME

# Navigate to project directory
cd /gpfs/home/acad/ulg-sysmod/lpirenne/delete_context

# Run the comparison script with the specified dataset
python -m context_deletion.all_comparisons --dataset $DATASET --num_samples 1000 --mode all --version v9 --output_dir ./context_deletion/results_v9_2 --alpha 0.5 --num_ablations $ABLATIONS --model_name Qwen/Qwen2.5-7B-Instruct-1M
EOF

