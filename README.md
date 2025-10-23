# Context Deletion Analysis Framework

This repository implements a comprehensive framework for analyzing contributive attribution in question answering systems. It compares three main approaches: **TreeFinder**, **TracLLM** and **ContextCite** across multiple benchmark datasets.

## Overview

The framework evaluates how different contributive attribution methods perform on long-context question answering tasks by computing necessity and sufficiency scores for individual sentences within the context. This analysis helps understand which parts of the context are used by LLMs for generating accurate answers.

### Key Methods

1. **TreeFinder**: A hierarchical approach that uses a tree-based structure to remove parts of the context in chunks and measures, with necessity and sufficiency scores, the answer score drop.
2. **ContextCite**: An ablation-based method that measures context importance through the measure of the answer probability under a multitude of random removals in the context.
3. **TracLLM**: A hierarchical approach that uses a tree-based structure to remove parts of the context in chunks and measures the answer probability drops for some perturbations, then use a model to predict the scores of all sentences at certain level in the tree.

TreeFinder works for sure using the Transformers variant. The others are provided as examples and might not lead to the same results or might be bugged.

## Project Structure

```
context_deletion/
├── all_comparisons.py      # Main comparison script
├── tree_finder.py          # TreeFinder implementation
├── tree_utils.py           # Utility classes and base implementations
└── slurm.sbatch.py         # Slurm script
```

## Requirements

The project requires the following main dependencies:

- **PyTorch** (with CUDA support recommended)
- **Transformers** (Hugging Face)
- **datasets** (Hugging Face datasets library)
- **NLTK** (for sentence tokenization)
- **matplotlib** (for visualization)
- **numpy**
- **tqdm** (for progress bars)
- **vLLM** (for efficient inference)

- **ContextCite**: This project imports `ContextCiter` from the `context_cite` module, which needs to be installed
- **TracLLM-kit**:  This project imports `TracLLM` from the `tracllmkit` module, which needs to be installed
- **GPU**: CUDA-compatible GPU recommended for efficient processing

## Datasets

The framework supports four main datasets:

- **hotpot_qa**: Multi-hop reasoning questions from HotpotQA
- **loogle_short**: Short dependency questions from LooGLE
- **loogle_long**: Long dependency questions from LooGLE  
- **longbench**: Long context benchmark from LongBench-v2

All datasets are automatically downloaded from Hugging Face datasets when first used.

To fit in our GPUs, we limited the size of the dataset samples to under 20 000 tokens. This has been left in for reproducibility of the results but it is entirely optional.

## Usage

### Basic Command Structure

```bash
python all_comparisons.py \
    --dataset <dataset_name> \
    --num_samples <number> \
    --mode <mode> \
    --version <version> \
    --output_dir <output_directory> \
    --alpha <alpha_value> \
    --num_ablations <ablations>
```

### Parameters

- `--dataset`: Choose from `hotpot_qa`, `loogle_short`, `loogle_long`, `longbench`
- `--num_samples`: Number of samples to process (default: 1000)
- `--mode`: Execution mode
    - Standard: `all`, `TreeFinder`, `ContextCite`, `TracLLM`
    - Results: `metrics`, `aggregate_plots`, `get_best_sentences`, `ranking_analysis`
    - Step: `necessity_sufficiency`, `topw_group_scores`
- `--output_dir`: Directory for output files (default: `./results`)
- `--alpha`: Weighting parameter for necessity vs sufficiency (0.0-1.0)
- `--version`: The version name for differentiating results (`v2` for alpha=0.25 in the paper, `a0` for alpha=0.0 in the paper, `a1` for alpha=1.0 in the paper)
- `--num_ablations`: Number of ablation for ContextCite (32, 50, 100)
- `--model_name`: Model to use (default: `Qwen/Qwen2.5-7B-Instruct-1M`)

### Execution Modes

1. **`all`**: Run complete pipeline (TreeFinder + ContextCite + metrics + plots)
2. **`TreeFinder`**: Run only TreeFinder analysis
3. **`ContextCite`**: Run only ContextCite analysis
4. **`TracLLM`**: Run only TracLLM analysis
5. **`metrics`**: Generate necessity/sufficiency metrics from existing results
6. **`aggregate_plots`**: Create comparison visualizations
7. **`get_best_sentences`**: Write in a text file the top 5 sentences for each method
8. **`ranking_analysis`**: Compare ranks of ground truth sentences as classified by each method
9. **`necessity_sufficiency`**: Compute both scores for each sentence in the datasets
10. **`topw_group_scores`**: Compute all scores per sentences in the top 5 as groups (1, 1-2, 1-3, 1-4 and 1-5)

### SLURM Execution

For HPC environments, use the provided SLURM script (complete it with your path, conda environment and account):

```bash
# Submit job for specific dataset
./slurm.sbatch hotpot_qa
./slurm.sbatch loogle_long
./slurm.sbatch longbench
```

The script automatically configures resources based on dataset requirements:
- **hotpot_qa**: 1 GPU, 24h time limit
- **loogle_short/long**: 2 GPUs, 48h time limit  
- **longbench**: 2 GPUs, 48h time limit

## Example Commands

### Complete Analysis Pipeline

```bash
# Run full analysis on HotpotQA
python all_comparisons.py --dataset hotpot_qa --num_samples 1000 --mode all --version v2 --output_dir ./results --alpha 0.25 --num_ablations 32
```

### Batch Processing Commands

```bash
# Process all datasets with different configurations
python all_comparisons.py --dataset loogle_short --num_samples 1000 --mode metrics --version v2 --output_dir ./results --alpha 0.25 --num_ablations 32
python all_comparisons.py --dataset loogle_short --num_samples 1000 --mode metrics --version v2 --output_dir ./results --alpha 0.25 --num_ablations 100
python all_comparisons.py --dataset loogle_long --num_samples 1000 --mode metrics --version v2 --output_dir ./results --alpha 0.25 --num_ablations 100
python all_comparisons.py --dataset loogle_long --num_samples 1000 --mode metrics --version v2 --output_dir ./results --alpha 0.25 --num_ablations 32
python all_comparisons.py --dataset loogle_long --num_samples 1000 --mode metrics --version a0 --output_dir ./results --alpha 0.0 --num_ablations 32
python all_comparisons.py --dataset loogle_long --num_samples 1000 --mode metrics --version a1 --output_dir ./results --alpha 1.0 --num_ablations 32
python all_comparisons.py --dataset longbench --num_samples 1000 --mode metrics --version v2 --output_dir ./results --alpha 0.25 --num_ablations 100
python all_comparisons.py --dataset longbench --num_samples 1000 --mode metrics --version v2 --output_dir ./results --alpha 0.25 --num_ablations 32
python all_comparisons.py --dataset longbench --num_samples 1000 --mode metrics --version a0 --output_dir ./results --alpha 0.0 --num_ablations 32
python all_comparisons.py --dataset longbench --num_samples 1000 --mode metrics --version a1 --output_dir ./results --alpha 1.0 --num_ablations 32
python all_comparisons.py --dataset hotpot_qa --num_samples 1000 --mode metrics --version v2 --output_dir ./results --alpha 0.25 --num_ablations 50
python all_comparisons.py --dataset hotpot_qa --num_samples 1000 --mode metrics --version v2 --output_dir ./results --alpha 0.25 --num_ablations 32
python all_comparisons.py --dataset hotpot_qa --num_samples 1000 --mode metrics --version a0 --output_dir ./results --alpha 0.0 --num_ablations 32
python all_comparisons.py --dataset hotpot_qa --num_samples 1000 --mode metrics --version a1 --output_dir ./results --alpha 1.0 --num_ablations 32
```

## Output Files

The framework generates several types of output files:

### JSON Results
- `{dataset}_ContextCite_{num_ablations}.json`: ContextCite attribution scores
- `{dataset}_TreeFinder_{version}.json`: TreeFinder attribution scores
- `{dataset}_TracLLM_{version}.json`: TracLLM attribution scores
- `{dataset}_necessity_sufficiency.json`: Necessity and sufficiency metrics
- `{dataset}_ranking_analysis_{version}.json`: Ranking comparison analysis
- `{dataset}_topk_group_scores_{version}_{num_ablations}.json`: Top-k group analysis

These JSON files are saved every 10 steps so that many of the scripts can simply be ran again if they have been interrupted. This can also be used to grow the num_samples.

This is not compatible with a change of filtering since the ids are not tracked. If you change the filtering process, delete your old save or change the version name.

### PDF Visualizations
- `{dataset}_top10_{version}_{num_ablations}_aggregate_comparison_three_metrics.pdf`: Comparison plots
- `{dataset}_topk_group_scores_{version}_{num_ablations}.pdf`: Score distribution plots

## Algorithm Versions

These are the conventions that have been used for the results the paper, they have no effect other than changing the save files' names.

- **v2**: Standard version with alpha=0.25 (balanced necessity/sufficiency)
- **a0**: Sufficiency-only version with alpha=0.0  
- **a1**: Necessity-only version with alpha=1.0

## Paper

This is the code that was used to run all the experiments of the paper: 
"Contributive Attribution for Question Answering via Tree-based Context Pruning" by Lize Pirenne, Gaspard Lambrechts, Norman Marlier, Maxence de la Brassine Bonardeaux, Gilles Louppe and Damien Ernst.

All code has been written by Lize Pirenne.

## Use of Generative AI

- Generative AI has been used to help writing the code and README (Github Copilot).
- The paper has been proof-read throughout its writing process for errors of language (Writefull).
