#!/bin/bash

# Usage: ./slurm.sbatch <dataset>
# Where dataset is one of: hotpot_qa, loogle_short, loogle_long, longbench

# Get dataset from command line argument
DATASET=${1:-hotpot_qa}

# Set resources based on dataset
case $DATASET in
    "hotpot_qa")
        GPUS=1
        TIME="1-00:00:00"
        ;;
    "loogle_short")
        GPUS=2
        TIME="2-00:00:00"
        ;;
    "loogle_long")
        GPUS=2
        TIME="2-00:00:00"
        ;;
    "longbench")
        GPUS=2
        TIME="2-00:00:00"
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
CPUS=$((8 * GPUS))
MEM=$((60 * GPUS))

# Submit the job with the appropriate parameters
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=TreeFinder_${DATASET}
#SBATCH --gpus=${GPUS}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --time=${TIME}
#SBATCH --mem=${MEM}GB
#SBATCH -o log_${DATASET}.out
#SBATCH -e log_${DATASET}.err
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
cd /path/to/your/project

# Run the comparison script with the specified dataset
python all_comparisons.py --dataset $DATASET --num_samples 1000 --mode all --version v2 --output_dir ./results --alpha 0.25 --num_ablations 100
EOF

