#!/usr/bin/env python3
"""
Comprehensive comparison script for TreeFinder vs ContextCite vs TracLLM methods.
"""

import argparse
from dataclasses import dataclass
import json
import logging
import traceback
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# Create custom legend
from matplotlib.patches import Patch


from context_cite import ContextCiter

from tracllmkit import PerturbationBasedAttribution

from tree_finder import TreeFinderTransformer
from tree_utils import DEFAULT_PROMPT_TEMPLATE, DEFAULT_GENERATE_KWARGS

import torch._dynamo

torch._dynamo.config.suppress_errors = True


class BaseContextPartitioner:
    """Base class for partitioning context into sources."""

    def __init__(self, context: str) -> None:
        self.context = context

    @property
    def num_sources(self) -> int:
        """The number of sources."""
        raise NotImplementedError

    def get_source(self, index: int) -> str:
        """Get a representation of the source corresponding to a given index."""
        raise NotImplementedError

    def get_context(self, mask=None):
        """Get a version of the context ablated according to the given mask."""
        raise NotImplementedError

    @property
    def sources(self) -> List[str]:
        """A list of all sources."""
        return [self.get_source(i) for i in range(self.num_sources)]


class CustomContextPartitioner(BaseContextPartitioner):
    """Custom partitioner that uses pre-split sentences."""

    def __init__(self, context: str, sentences: List[str]) -> None:
        super().__init__(context)
        self.sentences = sentences

    @property
    def num_sources(self) -> int:
        return len(self.sentences)

    def get_source(self, index: int) -> str:
        return self.sentences[index]

    def get_context(self, mask=None) -> str:
        if mask is None:
            return self.context
        return "".join(
            [self.sentences[i] for i in range(len(self.sentences)) if mask[i]]
        )


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_and_prepare_dataset(
    tokenizer, dataset_name: str = "hotpot_qa", num_samples: int = 100
):
    """Load and prepare the dataset for processing."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading {dataset_name} dataset...")

    def count_tokens(text):
        """Count tokens in a text string."""
        return len(tokenizer.encode(text, add_special_tokens=False))

    if dataset_name == "hotpot_qa":
        dataset = load_dataset(
            "hotpotqa/hotpot_qa", "fullwiki", split="train", trust_remote_code=True
        )
    elif dataset_name == "loogle_short":
        dataset = load_dataset("bigai-nlco/LooGLE", "shortdep_qa", split="test")
        dataset = dataset.filter(lambda x: count_tokens(x["context"]) < 20000)
    elif dataset_name == "loogle_long":
        dataset = load_dataset("bigai-nlco/LooGLE", "longdep_qa", split="test")
        dataset = dataset.filter(lambda x: count_tokens(x["context"]) < 20000)
    elif dataset_name == "longbench":
        dataset = load_dataset("THUDM/LongBench-v2", split="train")
        dataset = dataset.filter(lambda x: count_tokens(x["context"]) < 20000)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print(f"Dataset {dataset_name} loaded with {len(dataset)} samples.")

    # Take only the first num_samples
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    logger.info(f"Loaded {len(dataset)} samples")

    return dataset


def preprocess_sample(sample, dataset: str = "hotpot_qa"):
    """Preprocess a single sample from the dataset."""
    question = sample["question"]
    context = sample["context"]
    answer = sample["answer"]

    if dataset == "hotpot_qa":
        # Extract sentences from HotpotQA context format
        sentences = [
            sentence for title, sentence in context.items() if title != "title"
        ]
        sentences = [sentence for sublist in sentences for sentence in sublist]
        sentences = [sentence for sublist in sentences for sentence in sublist]
        context_str = " ".join(sentences)
    else:
        context_str = context

    sentences = sent_tokenize(context_str)

    return {
        "question": question,
        "context": context_str,
        "sentences": sentences,
        "answer": answer,
    }


def initialize_models(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", mode: str = "default"
):
    """Initialize the tokenizer and model."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model: {model_name}")

    # quantization_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_use_double_quant=True,
    #    bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"

    if mode != "metrics":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # Use bfloat16 for better performance
        )
        model.eval()

        model.generation_config.cache_implementation = "dynamic"
    else:
        model = None

    return model, tokenizer


def get_context_cite_scores(
    model,
    tokenizer,
    question: str,
    context: str,
    sentences: List[str],
    batch_size: int = 1,
    num_ablations: int = 32,
):
    """Get ContextCite attribution scores."""
    partitioner = CustomContextPartitioner(context=context, sentences=sentences)

    cc = ContextCiter(
        model=model,
        tokenizer=tokenizer,
        context=context,
        query=question,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        generate_kwargs=DEFAULT_GENERATE_KWARGS,
        num_ablations=num_ablations,
        batch_size=batch_size,
        partitioner=partitioner,
    )

    weight = cc.get_attributions(start_idx=0, end_idx=len(cc.response), verbose=False)
    return weight


def process_single_sample_tree_finder(
    sample_idx: int, sample_data: Dict[str, Any], finder
) -> Dict[str, Any]:
    """Process a single sample and return comparison results."""
    logger = logging.getLogger(__name__)

    question = sample_data["question"]
    sentences = sample_data["sentences"]

    # Get TreeFinder scores with timing
    try:
        finder.get_and_reset_num_calls()
        start_time = time.time()
        tree_finder_scores = finder.get_scores(question, sentences)
        tree_finder_time = time.time() - start_time
    except Exception as e:
        logger.error(f"Sample {sample_idx}: TreeFinder scores failed: {e}")
        return None

    tree_finder_num_calls = finder.get_and_reset_num_calls()

    return {
        "data_index": sample_idx,
        "scores": tree_finder_scores.tolist(),
        "num_calls": tree_finder_num_calls,
        "time": tree_finder_time,
    }


def process_single_sample_context_cite(
    sample_idx: int,
    sample_data: Dict[str, Any],
    model,
    tokenizer,
    num_ablations: int = 32,
    batch_size: int = 1,
) -> Dict[str, Any]:
    """Process a single sample and return comparison results."""
    logger = logging.getLogger(__name__)

    question = sample_data["question"]
    context = sample_data["context"]
    sentences = sample_data["sentences"]

    # Get ContextCite scores with timing
    try:
        start_time = time.time()
        context_cite_weights = get_context_cite_scores(
            model,
            tokenizer,
            question,
            context,
            sentences,
            num_ablations=num_ablations,
            batch_size=batch_size,
        )
        context_cite_time = time.time() - start_time
    except Exception as e:
        logger.error(f"Sample {sample_idx}: ContextCite failed: {e}")
        return None

    return {
        "data_index": sample_idx,
        "scores": context_cite_weights.tolist(),
        "num_calls": num_ablations,
        "time": context_cite_time,
    }


def evaluate_sentence_scores(
    sample_idx: int, sample_data: Dict[str, Any], finder
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)

    # Get necessity and sufficiency scores with error handling
    try:
        question = sample_data["question"]
        sentences = sample_data["sentences"]

        necessity, sufficiency = finder.get_necessity_sufficiency(
            question,
            sentences,
            indices=[
                [i] for i in range(len(sentences))
            ],  # Each sentence as a separate index,
        )

    except Exception as e:
        logger.error(f"Sample {sample_idx}: Necessity/sufficiency failed: {e}")
        return None

    return {
        "data_index": sample_idx,
        "necessity_scores": necessity.tolist(),
        "sufficiency_scores": sufficiency.tolist(),
    }


def create_aggregate_plots(
    results: List[Dict[str, Any]], output_dir: Path, dataset: str
):
    """Create three aggregate comparison plots with box plots for necessity, sufficiency, and combined."""
    logger = logging.getLogger(__name__)
    logger.info("Creating aggregate comparison plots...")

    # Calculate average timing for both methods
    avg_tree_finder_time = np.mean(
        [r["TreeFinder_time"] for r in results if r is not None]
    )
    avg_context_cite_time = np.mean(
        [r["ContextCite_time"] for r in results if r is not None]
    )
    avg_tracllm_time = np.mean([r["TracLLM_time"] for r in results if r is not None])
    avg_tree_finder_calls = np.mean(
        [r["TreeFinder_num_calls"] for r in results if r is not None]
    )
    avg_context_cite_calls = np.mean(
        [r["ContextCite_num_calls"] for r in results if r is not None]
    )
    avg_tracllm_calls = np.mean(
        [r["TracLLM_num_calls"] for r in results if r is not None]
    )

    # Find the maximum length across all samples
    max_length = max(
        len(result["necessity_scores"]) for result in results if result is not None
    )

    # Initialize data collection for statistics
    metrics = ["necessity", "sufficiency", "combined"]
    tree_finder_data = {metric: [[] for _ in range(max_length)] for metric in metrics}
    context_cite_data = {metric: [[] for _ in range(max_length)] for metric in metrics}
    ground_truth_data = {metric: [[] for _ in range(max_length)] for metric in metrics}
    tracllm_data = {metric: [[] for _ in range(max_length)] for metric in metrics}

    for result in results:
        if result is None:
            continue

        for metric in metrics:
            tf_scores = result[f"TreeFinder_{metric}_scores"]
            cc_scores = result[f"ContextCite_{metric}_scores"]
            tl_scores = result[f"TracLLM_{metric}_scores"]
            gt_scores = result[f"{metric}_scores"]

            # Collect data for each position
            for i in range(len(tf_scores)):
                tree_finder_data[metric][i].append(tf_scores[i])
                context_cite_data[metric][i].append(cc_scores[i])
                ground_truth_data[metric][i].append(gt_scores[i])
                tracllm_data[metric][i].append(tl_scores[i])

    # Create three separate plots
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for idx, metric in enumerate(metrics):
        # Prepare data for box plots
        box_data = []
        labels = []
        colors = []
        positions = []

        pos_counter = 1
        for i in range(max_length):
            if tree_finder_data[metric][i]:  # If we have data for this position
                # Filter out NaN values
                tf_values = np.array(tree_finder_data[metric][i])
                cc_values = np.array(context_cite_data[metric][i])
                tl_values = np.array(tracllm_data[metric][i])
                gt_values = np.array(ground_truth_data[metric][i])

                tf_values = tf_values[~np.isnan(tf_values)]
                cc_values = cc_values[~np.isnan(cc_values)]
                tl_values = tl_values[~np.isnan(tl_values)]
                gt_values = gt_values[~np.isnan(gt_values)]

                # Add data for each method at this position
                box_data.extend([tf_values, cc_values, tl_values, gt_values])
                labels.extend([f"TF{i + 1}", f"CC{i + 1}", f"GT{i + 1}", f"TL{i + 1}"])
                colors.extend(["lightblue", "lightcoral", "plum", "lightgreen"])
                positions.extend(
                    [pos_counter, pos_counter + 1, pos_counter + 2, pos_counter + 3]
                )
                pos_counter += 5  # Space between groups

        # Create box plot
        bp = axes[idx].boxplot(
            box_data,
            positions=positions,
            patch_artist=True,
            showfliers=True,
            widths=0.8,
            whis=[1, 99],
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Style the whiskers to be lighter and thinner
        for whisker in bp["whiskers"]:
            whisker.set_color("gray")
            whisker.set_alpha(0.5)
            whisker.set_linewidth(1.0)

        # Style the caps (whisker ends) to be lighter and thinner
        for cap in bp["caps"]:
            cap.set_color("gray")
            cap.set_alpha(0.5)
            cap.set_linewidth(1.0)

        # Style the outliers (fliers) to be lighter and smaller
        for flier in bp["fliers"]:
            flier.set_marker("o")
            flier.set_markerfacecolor("gray")
            flier.set_markeredgecolor("gray")
            flier.set_alpha(0.3)
            flier.set_markersize(3)

        legend_elements = [
            Patch(facecolor="lightblue", alpha=0.7, label="TreeFinder"),
            Patch(facecolor="lightcoral", alpha=0.7, label="ContextCite"),
            Patch(facecolor="plum", alpha=0.7, label="TracLLM"),
            Patch(facecolor="lightgreen", alpha=0.7, label="Ground Truth"),
        ]

        axes[idx].set_title(f"{metric.capitalize()} Scores", fontsize=26)
        if idx == 1:
            axes[idx].set_xlabel("Sentence Index (Ordered by Score)", fontsize=22)
        elif idx == 0:
            axes[idx].set_ylabel("Score", fontsize=22)
        axes[idx].legend(handles=legend_elements, fontsize=22, loc="upper right")
        axes[idx].grid(True, alpha=0.3)

        # Set x-axis labels
        group_centers = [
            np.mean(positions[i : i + 4]) for i in range(0, len(positions), 4)
        ]
        axes[idx].set_xticks(group_centers)
        axes[idx].set_xticklabels(
            [str(i + 1) for i in range(len(group_centers))], rotation=0
        )

        # Increase font size for better readability
        axes[idx].tick_params(axis="both", which="major", labelsize=18)
        axes[idx].tick_params(axis="both", which="minor", labelsize=16)

        # logarithmic scale for better visibility
        axes[idx].set_yscale("log")

    # Add timing information at the bottom
    timing_text = (
        f"Average Time - TreeFinder: {avg_tree_finder_time:.2f}s "
        f"(Avg calls: {avg_tree_finder_calls:.1f}), "
        f"ContextCite: {avg_context_cite_time:.2f}s "
        f"(Avg calls: {avg_context_cite_calls:.1f}), "
        f"TracLLM: {avg_tracllm_time:.2f}s "
        f"(Avg calls: {avg_tracllm_calls:.1f})"
    )
    fig.text(
        0.5,
        0.02,
        timing_text,
        ha="center",
        fontsize=20,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )

    plt.tight_layout(pad=4.0, w_pad=0.0, h_pad=4.0)

    # Save plot
    plot_path = output_dir / f"{dataset}_aggregate_comparison_three_metrics.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Aggregate plots saved to {plot_path}")
    logger.info(
        f"Timing - TreeFinder: {avg_tree_finder_time:.2f}s (avg {avg_tree_finder_calls:.1f} calls), ContextCite: {avg_context_cite_time:.2f}s"
    )


def convert_tensors_to_lists(obj):
    """Recursively convert tensors to lists for JSON serialization."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    elif hasattr(obj, "cpu"):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_tensors_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_lists(item) for item in obj]
    else:
        return obj


def save_incremental_results(results: List[Dict], output_path: Path, logger):
    """Save results incrementally to JSON file."""
    json_safe_results = convert_tensors_to_lists(results)
    with open(output_path, "w") as f:
        json.dump(json_safe_results, f, indent=2)
    logger.info(f"Saved {len(results)} results to {output_path}")


def load_existing_results(output_path: Path, logger) -> tuple[List[Dict], set[int]]:
    """Load existing results if they exist and return results list and set of processed indices."""
    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                results = json.load(f)
            processed_indices = {
                result["data_index"] for result in results if "data_index" in result
            }
            logger.info(
                f"Loaded {len(results)} existing results from {output_path}, covering indices: {sorted(processed_indices)}"
            )
            return results, processed_indices
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")
            return [], set()
    return [], set()


def compute_combined_scores(
    necessity_scores: List[float], sufficiency_scores: List[float], alpha: float = 0.5
) -> List[float]:
    """Compute combined scores from necessity and sufficiency scores."""
    return [
        alpha * n + (1 - alpha) * s
        for n, s in zip(necessity_scores, sufficiency_scores)
    ]


def run_tree_finder_mode(args: Dict, logger, dataset, model, tokenizer) -> List[Dict]:
    """Execute tree finder mode."""
    logger.info("Running TreeFinder mode...")

    output_path = (
        Path(args["output_dir"])
        / f"{args['dataset']}_TreeFinder_{args['version']}.json"
    )
    print(f"Output path for TreeFinder results: {output_path}")
    results, processed_indices = load_existing_results(output_path, logger)

    # Initialize TreeFinder
    finder = TreeFinderTransformer(
        model_name=args["model_name"],
        model_path=args["model_path"],
        model=model,
        tokenizer=tokenizer,
        threshold=args["threshold"],
        factor=args["factor"],
        cum_prob=args["cum_prob"],
        topk=args["topk"],
        batch_size=args["batch_size"],
        logger=logger,
        alpha=args["alpha"],
    )

    for idx in tqdm(range(len(dataset)), desc="Processing TreeFinder"):
        if idx in processed_indices:
            continue

        try:
            sample = dataset[idx]
            processed_sample = preprocess_sample(sample, args["dataset"])

            result = process_single_sample_tree_finder(idx, processed_sample, finder)
            if result:
                results.append(result)

            # Save every 10 samples
            if (len(results) % 10 == 0) or (idx == len(dataset) - 1):
                save_incremental_results(results, output_path, logger)

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def run_context_cite_mode(args: Dict, logger, dataset, model, tokenizer) -> List[Dict]:
    """Execute context cite mode."""
    logger.info("Running ContextCite mode...")

    output_path = (
        Path(args["output_dir"])
        / f"{args['dataset']}_ContextCite_{args['num_ablations']}.json"
    )
    print(f"Output path for ContextCite results: {output_path}")
    results, processed_indices = load_existing_results(output_path, logger)

    for idx in tqdm(range(len(dataset)), desc="Processing ContextCite"):
        if idx in processed_indices:
            continue

        try:
            sample = dataset[idx]
            processed_sample = preprocess_sample(sample, args["dataset"])

            result = process_single_sample_context_cite(
                idx,
                processed_sample,
                model,
                tokenizer,
                num_ablations=args["num_ablations"],
                batch_size=args["batch_size"],
            )
            if result:
                results.append(result)

            # Save every 10 samples
            if (len(results) % 10 == 0) or (idx == len(dataset) - 1):
                save_incremental_results(results, output_path, logger)

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def run_tracllm_mode(args: Dict, logger, dataset, model, tokenizer) -> List[Dict]:
    """Execute TracLLM mode."""
    logger.info("Running TracLLM mode...")

    output_path = (
        Path(args["output_dir"]) / f"{args['dataset']}_TracLLM_{args['version']}.json"
    )
    results, processed_indices = load_existing_results(output_path, logger)

    for idx in tqdm(range(len(dataset)), desc="Processing TracLLM"):
        if idx in processed_indices:
            continue

        try:
            sample = dataset[idx]
            processed_sample = preprocess_sample(sample, args["dataset"])

            question = processed_sample["question"]
            sentences = processed_sample["sentences"]

            # create a wrapper so that we can call llm.model, llm.tokenizer, llm.name
            @dataclass
            class llm_wrapper:
                model: Any
                tokenizer: Any

                @property
                def name(self):
                    return args["model_name"]

            # Get TracLLM scores with timing
            try:
                trac_llm = PerturbationBasedAttribution(
                    llm=llm_wrapper(model, tokenizer),
                    explanation_level="sentence",
                    attr_type="tracllm",
                    K=6,
                )

                start_time = time.time()

                prompt = DEFAULT_PROMPT_TEMPLATE.format(
                    query=question, context=" ".join(sentences)
                )

                message = [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]

                tokenized_chat = tokenizer.apply_chat_template(
                    message,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    enable_thinking=False,
                )

                outputs = model.generate(
                    tokenized_chat.to(model.device),
                    return_dict_in_generate=True,
                    output_scores=False,
                    **DEFAULT_GENERATE_KWARGS,
                )
                generated_token_ids = outputs.sequences[0, tokenized_chat.shape[1] :]
                answer = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

                text, important_ids, important_scores, _, _ = trac_llm.attribute(
                    question, sentences, answer
                )
                # pad the rest of the scores with -inf
                all_scores = [-float("inf")] * len(sentences)
                for i, score in zip(important_ids, important_scores):
                    all_scores[i] = score
                important_scores = all_scores
                trac_llm_time = time.time() - start_time
            except Exception as e:
                logger.error(f"Sample {idx}: TracLLM scores failed: {e}")
                traceback.print_exc()
                continue

            result = {
                "data_index": idx,
                "scores": important_scores,
                "num_calls": trac_llm.num_calls,
                "time": trac_llm_time,
            }
            results.append(result)
            trac_llm.num_calls = 0  # reset for next sample

            # Save every 10 samples
            if (len(results) % 10 == 0) or (idx == len(dataset) - 1):
                save_incremental_results(results, output_path, logger)

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def run_necessity_sufficiency_mode(
    args: Dict, logger, dataset, model, tokenizer
) -> List[Dict]:
    """Execute necessity and sufficiency mode."""
    logger.info("Running Necessity/Sufficiency mode...")

    output_path = (
        Path(args["output_dir"]) / f"{args['dataset']}_necessity_sufficiency.json"
    )
    results, processed_indices = load_existing_results(output_path, logger)

    # Initialize TreeFinder for necessity/sufficiency computation
    finder = TreeFinderTransformer(
        model_name=args["model_name"],
        model_path=args["model_path"],
        model=model,
        tokenizer=tokenizer,
        threshold=args["threshold"],
        factor=args["factor"],
        cum_prob=args["cum_prob"],
        topk=args["topk"],
        batch_size=args["batch_size"],
        logger=logger,
        alpha=args["alpha"],
    )

    for idx in tqdm(range(len(dataset)), desc="Processing Necessity/Sufficiency"):
        if idx in processed_indices:
            continue

        try:
            sample = dataset[idx]
            processed_sample = preprocess_sample(sample, args["dataset"])

            result = evaluate_sentence_scores(idx, processed_sample, finder)
            if result:
                results.append(result)

            # Save every 10 samples
            if (len(results) % 10 == 0) or (idx == len(dataset) - 1):
                save_incremental_results(results, output_path, logger)

        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def run_aggregate_plots_mode(args: Dict, logger) -> None:
    """Create aggregate plots from existing results."""
    logger.info("Creating aggregate plots...")

    output_dir = Path(args["output_dir"])

    # Load all required results
    tree_finder_path = (
        output_dir / f"{args['dataset']}_TreeFinder_{args['version']}.json"
    )
    context_cite_path = (
        output_dir / f"{args['dataset']}_ContextCite_{args['num_ablations']}.json"
    )
    tracllm_path = output_dir / f"{args['dataset']}_TracLLM_{args['version']}.json"
    necessity_path = output_dir / f"{args['dataset']}_necessity_sufficiency.json"

    if not all(
        path.exists()
        for path in [tree_finder_path, context_cite_path, necessity_path, tracllm_path]
    ):
        logger.error(
            "Missing required result files for aggregate plots: "
            f"{tree_finder_path}, {context_cite_path}, {necessity_path}"
        )
        return

    tree_finder_results, _ = load_existing_results(tree_finder_path, logger)
    context_cite_results, _ = load_existing_results(context_cite_path, logger)
    necessity_results, _ = load_existing_results(necessity_path, logger)
    tracllm_results, _ = load_existing_results(tracllm_path, logger)

    # Combine results by matching data indices and computing top 10 by combined scores
    combined_results = []

    for ns_result in necessity_results:
        data_idx = ns_result["data_index"]

        # Find matching results
        cc_result = next(
            (r for r in context_cite_results if r["data_index"] == data_idx), None
        )
        tf_result = next(
            (r for r in tree_finder_results if r["data_index"] == data_idx), None
        )
        tl_result = next(
            (r for r in tracllm_results if r["data_index"] == data_idx), None
        )

        if cc_result and tf_result and tl_result:
            # Compute combined scores for ranking
            combined_score = compute_combined_scores(
                ns_result["necessity_scores"],
                ns_result["sufficiency_scores"],
                alpha=0.5,
            )

            # Get top 10 sentences by combined score
            sentence_scores = list(zip(range(len(combined_score)), combined_score))
            sentence_scores.sort(key=lambda x: x[1], reverse=False)
            top_10_indices = [idx for idx, _ in sentence_scores[:10]]

            # Get top 10 indices for tree finder and context cite
            tf_top_10_indices = sorted(
                range(len(tf_result["scores"])),
                key=lambda i: tf_result["scores"][i],
                reverse=False,
            )[:10]
            cc_top_10_indices = sorted(
                range(len(cc_result["scores"])),
                key=lambda i: cc_result["scores"][i],
                reverse=True,
            )[:10]
            tl_result_top_10_indices = sorted(
                range(len(tl_result["scores"])),
                key=lambda i: tl_result["scores"][i],
                reverse=True,
            )[:10]

            combined_result = {
                "data_index": data_idx,
                "TreeFinder_necessity_scores": [
                    ns_result["necessity_scores"][i] for i in tf_top_10_indices
                ],
                "TreeFinder_sufficiency_scores": [
                    ns_result["sufficiency_scores"][i] for i in tf_top_10_indices
                ],
                "ContextCite_necessity_scores": [
                    ns_result["necessity_scores"][i] for i in cc_top_10_indices
                ],
                "ContextCite_sufficiency_scores": [
                    ns_result["sufficiency_scores"][i] for i in cc_top_10_indices
                ],
                "TracLLM_necessity_scores": [
                    ns_result["necessity_scores"][i] for i in tl_result_top_10_indices
                ],
                "TracLLM_sufficiency_scores": [
                    ns_result["sufficiency_scores"][i] for i in tl_result_top_10_indices
                ],
                "necessity_scores": [
                    ns_result["necessity_scores"][i] for i in top_10_indices
                ],
                "sufficiency_scores": [
                    ns_result["sufficiency_scores"][i] for i in top_10_indices
                ],
                "TreeFinder_combined_scores": [
                    combined_score[i] for i in tf_top_10_indices
                ],
                "ContextCite_combined_scores": [
                    combined_score[i] for i in cc_top_10_indices
                ],
                "TracLLM_combined_scores": [
                    combined_score[i] for i in tl_result_top_10_indices
                ],
                "combined_scores": [combined_score[i] for i in top_10_indices],
                "num_sentences": len(combined_score),
                "TreeFinder_time": tf_result["time"],
                "ContextCite_time": cc_result["time"],
                "TracLLM_time": tl_result["time"],
                "TreeFinder_num_calls": tf_result["num_calls"],
                "ContextCite_num_calls": cc_result["num_calls"],
                "TracLLM_num_calls": tl_result["num_calls"],
            }
            combined_results.append(combined_result)

    if combined_results:
        create_aggregate_plots(
            combined_results,
            output_dir,
            f"{args['dataset']}_top10_{args['version']}_{args['num_ablations']}",
        )
    else:
        logger.warning("No matching results found for aggregate plots")


def run_ranking_analysis_mode(args: Dict, logger) -> None:
    """Analyze ranking differences between methods."""
    logger.info("Running ranking analysis...")

    output_dir = Path(args["output_dir"])

    # Load all required results
    tree_finder_path = (
        output_dir / f"{args['dataset']}_TreeFinder_{args['version']}.json"
    )
    context_cite_path = (
        output_dir / f"{args['dataset']}_ContextCite_{args['num_ablations']}.json"
    )
    necessity_path = output_dir / f"{args['dataset']}_necessity_sufficiency.json"

    if not all(
        path.exists() for path in [tree_finder_path, context_cite_path, necessity_path]
    ):
        logger.error(
            "Missing required result files for ranking analysis: "
            f"{tree_finder_path}, {context_cite_path}, {necessity_path}"
        )
        return

    tree_finder_results, _ = load_existing_results(tree_finder_path, logger)
    context_cite_results, _ = load_existing_results(context_cite_path, logger)
    necessity_results, _ = load_existing_results(necessity_path, logger)

    # Analyze rankings - separate k=1 to k=5
    tf_positions_by_k = {k: [] for k in range(1, 6)}
    cc_positions_by_k = {k: [] for k in range(1, 6)}

    for tf_result in tree_finder_results:
        data_idx = tf_result["data_index"]

        # Find matching results
        cc_result = next(
            (r for r in context_cite_results if r["data_index"] == data_idx), None
        )
        ns_result = next(
            (r for r in necessity_results if r["data_index"] == data_idx), None
        )

        if cc_result and ns_result:
            # Get combined (ground truth) ranking
            combined_score = compute_combined_scores(
                ns_result["necessity_scores"],
                ns_result["sufficiency_scores"],
                alpha=0.5,
            )

            # Create rankings (indices sorted by score, descending)
            combined_ranking = sorted(
                range(len(combined_score)),
                key=lambda i: combined_score[i],
                reverse=False,
            )
            tf_scores = tf_result["scores"]
            cc_scores = cc_result["scores"]
            tf_ranking = sorted(
                range(len(tf_scores)), key=lambda i: tf_scores[i], reverse=False
            )
            cc_ranking = sorted(
                range(len(cc_scores)), key=lambda i: cc_scores[i], reverse=True
            )

            # Find positions of top k combined sentences in other rankings for k=1 to 5
            for k in range(1, 6):
                if len(combined_ranking) >= k:
                    topk_combined = combined_ranking[:k]

                    # Find position of each top-w sentence in the other methods
                    tf_positions = [
                        tf_ranking.index(sent_idx) + 1
                        for sent_idx in topk_combined
                        if sent_idx < len(tf_ranking)
                    ]
                    cc_positions = [
                        cc_ranking.index(sent_idx) + 1
                        for sent_idx in topk_combined
                        if sent_idx < len(cc_ranking)
                    ]

                    if tf_positions:
                        tf_positions_by_k[k].extend(tf_positions)
                    if cc_positions:
                        cc_positions_by_k[k].extend(cc_positions)

    # Compute and print metrics for each k
    if any(tf_positions_by_k[k] for k in range(1, 6)) and any(
        cc_positions_by_k[k] for k in range(1, 6)
    ):
        # Calculate total number of sentences across all samples
        total_sentences = sum(
            len(result["necessity_scores"])
            for result in necessity_results
            if any(r["data_index"] == result["data_index"] for r in tree_finder_results)
            and any(
                r["data_index"] == result["data_index"] for r in context_cite_results
            )
        )

        # Calculate number of samples with matching results
        num_samples = len(
            [
                result
                for result in necessity_results
                if any(
                    r["data_index"] == result["data_index"] for r in tree_finder_results
                )
                and any(
                    r["data_index"] == result["data_index"]
                    for r in context_cite_results
                )
            ]
        )

        # Calculate sentence statistics
        sentence_counts = [
            len(result["necessity_scores"])
            for result in necessity_results
            if any(r["data_index"] == result["data_index"] for r in tree_finder_results)
            and any(
                r["data_index"] == result["data_index"] for r in context_cite_results
            )
        ]

        avg_sentences = total_sentences / num_samples if num_samples > 0 else 0
        median_sentences = np.median(sentence_counts) if sentence_counts else 0

        logger.info("Ranking Analysis Results:")
        logger.info(f"Total sentences analyzed across all samples: {total_sentences}")
        logger.info(f"Number of samples: {num_samples}")
        logger.info(f"Average sentences per sample: {avg_sentences:.2f}")
        logger.info(f"Median sentences per sample: {median_sentences:.2f}")

        ranking_results = {}

        for k in range(1, 6):
            if tf_positions_by_k[k] and cc_positions_by_k[k]:
                avg_tf_position = np.median(tf_positions_by_k[k])
                avg_cc_position = np.median(cc_positions_by_k[k])

                logger.info(
                    f"k={k}: Median position of top-{k} combined sentences in TreeFinder ranking: {avg_tf_position:.2f}"
                )
                logger.info(
                    f"k={k}: Median position of top-{k} combined sentences in ContextCite ranking: {avg_cc_position:.2f}"
                )
                logger.info(
                    f"k={k}: TreeFinder advantage: {avg_cc_position - avg_tf_position:.2f} positions better on average"
                )

                ranking_results[f"k{k}_TreeFinder_avg_position"] = avg_tf_position
                ranking_results[f"k{k}_ContextCite_avg_position"] = avg_cc_position
                ranking_results[f"k{k}_TreeFinder_positions"] = tf_positions_by_k[k]
                ranking_results[f"k{k}_ContextCite_positions"] = cc_positions_by_k[k]

        ranking_path = (
            output_dir / f"{args['dataset']}_ranking_analysis_{args['version']}.json"
        )
        with open(ranking_path, "w") as f:
            json.dump(ranking_results, f, indent=2)
        logger.info(f"Ranking analysis saved to {ranking_path}")
    else:
        logger.warning("No valid ranking data found")


def run_topw_group_scores_mode(args: Dict, dataset, logger) -> None:
    """Compute and plot necessity/sufficiency for top-w groups of each method."""
    logger.info("Running top-w group necessity/sufficiency mode...")
    output_dir = Path(args["output_dir"])
    k_max = 5

    # Load all required results
    tree_finder_path = (
        output_dir / f"{args['dataset']}_TreeFinder_{args['version']}.json"
    )
    context_cite_path = (
        output_dir / f"{args['dataset']}_ContextCite_{args['num_ablations']}.json"
    )
    trac_llm_path = output_dir / f"{args['dataset']}_TracLLM_{args['version']}.json"
    necessity_path = output_dir / f"{args['dataset']}_necessity_sufficiency.json"

    if not all(
        path.exists() for path in [tree_finder_path, context_cite_path, necessity_path]
    ):
        logger.error(
            "Missing required result files for top-w group scores: "
            f"{tree_finder_path}, context_cite_path, necessity_path, trac_llm_path"
        )
        return

    tree_finder_results, _ = load_existing_results(tree_finder_path, logger)
    context_cite_results, _ = load_existing_results(context_cite_path, logger)
    tracllm_results, _ = load_existing_results(trac_llm_path, logger)
    necessity_results, _ = load_existing_results(necessity_path, logger)

    # Load existing top-w group scores if they exist
    output_path = (
        output_dir
        / f"{args['dataset']}_topk_group_scores_{args['version']}_{args['num_ablations']}.json"
    )
    all_group_scores, processed_indices = load_existing_results(output_path, logger)

    # Initialize TreeFinder for necessity/sufficiency computation
    model, tokenizer = initialize_models(args["model_name"], "default")
    finder = TreeFinderTransformer(
        model_name=args["model_name"],
        model_path=args["model_path"],
        model=model,
        tokenizer=tokenizer,
        threshold=args["threshold"],
        factor=args["factor"],
        cum_prob=args["cum_prob"],
        topk=args["topk"],
        batch_size=args["batch_size"],
        logger=logger,
        alpha=args["alpha"],
    )

    logger.info(
        f"Processing {len(necessity_results)} necessity results for top-w groups..."
    )

    for sample_idx in tqdm(
        range(len(necessity_results)), desc="Processing top-w groups"
    ):
        ns_result = necessity_results[sample_idx]
        data_idx = ns_result["data_index"]

        # Skip if already processed
        if data_idx in processed_indices:
            # check which method we have already processed
            if all(
                method in all_group_scores[sample_idx]
                for method in ["TreeFinder", "ContextCite", "Ground Truth", "TracLLM"]
            ):
                continue

        try:
            tf_result = next(
                (r for r in tree_finder_results if r["data_index"] == data_idx), None
            )
            cc_result = next(
                (r for r in context_cite_results if r["data_index"] == data_idx), None
            )
            tl_result = next(
                (r for r in tracllm_results if r["data_index"] == data_idx), None
            )

            if not tf_result or not cc_result or not tl_result:
                logger.warning(
                    f"Missing TreeFinder or ContextCite or TracLLM result for data index {data_idx}: skipping."
                )
                continue

            processed_sample = preprocess_sample(dataset[data_idx], args["dataset"])
            sentences = processed_sample["sentences"]
            question = processed_sample["question"]

            n_sent = len(sentences)

            tf_scores = tf_result["scores"]
            cc_scores = cc_result["scores"]
            tl_scores = tl_result["scores"]
            tf_ranking = sorted(
                range(n_sent), key=lambda i: tf_scores[i], reverse=False
            )
            cc_ranking = sorted(range(n_sent), key=lambda i: cc_scores[i], reverse=True)
            tl_ranking = sorted(range(n_sent), key=lambda i: tl_scores[i], reverse=True)
            gt_combined = compute_combined_scores(
                ns_result["necessity_scores"],
                ns_result["sufficiency_scores"],
                alpha=0.5,
            )
            gt_ranking = sorted(
                range(n_sent), key=lambda i: gt_combined[i], reverse=False
            )
            group_scores = {
                "data_index": data_idx,
                "TreeFinder": {},
                "ContextCite": {},
                "TracLLM": {},
                "Ground Truth": {},
            }

            for method, ranking in zip(
                ["TreeFinder", "ContextCite", "Ground Truth", "TracLLM"],
                [tf_ranking, cc_ranking, gt_ranking, tl_ranking],
            ):
                # Build all masks for this method
                solo_matrix = []

                for k in range(2, min(k_max, n_sent) + 1):
                    solo_matrix.append(ranking[:k])

                # Call get_necessity_sufficiency once with all masks
                necessity, sufficiency = finder.get_necessity_sufficiency(
                    question, sentences, solo_matrix
                )

                # Map results back to k values
                for k in range(2, min(k_max, n_sent) + 1):
                    group_scores[method][f"k={k}"] = {
                        "necessity": float(necessity[k - 2]),
                        "sufficiency": float(sufficiency[k - 2]),
                    }

                group_scores[method]["k=1"] = {
                    "necessity": float(ns_result["necessity_scores"][ranking[0]]),
                    "sufficiency": float(ns_result["sufficiency_scores"][ranking[0]]),
                }
            all_group_scores.append(group_scores)

            # Save every 10 samples
            if (len(all_group_scores) % 10 == 0) or (
                sample_idx == len(necessity_results) - 1
            ):
                save_incremental_results(all_group_scores, output_path, logger)

        except Exception as e:
            logger.error(f"Error processing sample {sample_idx}: {e}")
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save final results
    save_incremental_results(all_group_scores, output_path, logger)
    logger.info(f"Saved top-w group scores to {output_path}")


def run_get_best_sentences_mode(args: Dict, logger, dataset) -> None:
    """Loads the top 5 sentences from each method and ground truth and saves them to a text file."""
    logger.info("Running get best sentences mode...")

    output_dir = Path(args["output_dir"])

    # Load all required results
    tree_finder_path = (
        output_dir / f"{args['dataset']}_TreeFinder_{args['version']}.json"
    )
    context_cite_path = (
        output_dir / f"{args['dataset']}_ContextCite_{args['num_ablations']}.json"
    )
    necessity_path = output_dir / f"{args['dataset']}_necessity_sufficiency.json"
    trac_llm_path = output_dir / f"{args['dataset']}_TracLLM_{args['version']}.json"

    if not all(
        path.exists() for path in [tree_finder_path, context_cite_path, necessity_path]
    ):
        logger.error(
            "Missing required result files for get best sentences: "
            f"{tree_finder_path}, {context_cite_path}, {necessity_path}, {trac_llm_path}"
        )
        return

    tree_finder_results, _ = load_existing_results(tree_finder_path, logger)
    context_cite_results, _ = load_existing_results(context_cite_path, logger)
    necessity_results, _ = load_existing_results(necessity_path, logger)
    tracllm_results, _ = load_existing_results(trac_llm_path, logger)

    output_path = (
        output_dir
        / f"{args['dataset']}_best_sentences_{args['version']}_{args['num_ablations']}.txt"
    )

    with open(output_path, "w") as f:
        for ns_result in necessity_results:
            data_idx = ns_result["data_index"]

            tf_result = next(
                (r for r in tree_finder_results if r["data_index"] == data_idx), None
            )
            cc_result = next(
                (r for r in context_cite_results if r["data_index"] == data_idx), None
            )
            tl_result = next(
                (r for r in tracllm_results if r["data_index"] == data_idx), None
            )

            if not tf_result or not cc_result or not tl_result:
                logger.warning(
                    f"Missing TreeFinder or ContextCite or TracLLM result for data index {data_idx}: skipping."
                )
                continue

            processed_sample = preprocess_sample(dataset[data_idx], args["dataset"])
            sentences = processed_sample["sentences"]
            question = processed_sample["question"]
            n_sent = len(sentences)
            tf_scores = tf_result["scores"]
            cc_scores = cc_result["scores"]
            tl_scores = tl_result["scores"]
            tf_ranking = sorted(
                range(n_sent), key=lambda i: tf_scores[i], reverse=False
            )
            cc_ranking = sorted(range(n_sent), key=lambda i: cc_scores[i], reverse=True)
            tl_ranking = sorted(range(n_sent), key=lambda i: tl_scores[i], reverse=True)
            gt_combined = compute_combined_scores(
                ns_result["necessity_scores"],
                ns_result["sufficiency_scores"],
                alpha=0.5,
            )
            gt_ranking = sorted(
                range(n_sent), key=lambda i: gt_combined[i], reverse=False
            )

            f.write(f"Data Index: {data_idx}\n")
            f.write(f"Question: {question}\n\n")
            for method, ranking in zip(
                ["TreeFinder", "ContextCite", "Ground Truth", "TracLLM"],
                [tf_ranking, cc_ranking, gt_ranking, tl_ranking],
            ):
                f.write(f"Top 5 sentences from {method}:\n")
                for rank in range(min(5, len(ranking))):
                    sent_idx = ranking[rank]
                    f.write(
                        f"Rank {rank + 1} (Score: "
                        f"{tf_scores[sent_idx] if method == 'TreeFinder' else ''}"
                        f"{cc_scores[sent_idx] if method == 'ContextCite' else ''}"
                        f"{gt_combined[sent_idx] if method == 'Ground Truth' else ''}"
                        f"{tl_scores[sent_idx] if method == 'TracLLM' else ''}): "
                        f"{sentences[sent_idx]}\n"
                    )
                f.write("\n")
            f.write("=" * 80 + "\n\n")

    logger.info(f"Saved best sentences to {output_path}")


def plot_topw_group_scores(args: Dict, logger) -> None:
    """Plot necessity and sufficiency for top-w groups using box plots."""

    output_dir = Path(args["output_dir"])
    json_path = (
        output_dir
        / f"{args['dataset']}_topk_group_scores_{args['version']}_{args['num_ablations']}.json"
    )

    with open(json_path, "r") as f:
        all_group_scores = json.load(f)
    methods = ["TreeFinder", "ContextCite", "TracLLM", "Ground Truth"]
    k_vals = None
    # Collect data
    method_scores = {m: {"necessity": [], "sufficiency": []} for m in methods}
    for group in all_group_scores:
        for m in methods:
            k_keys = sorted(group[m].keys(), key=lambda x: int(x.split("=")[1]))
            if k_vals is None:
                k_vals = [int(x.split("=")[1]) for x in k_keys]
            for i, k in enumerate(k_keys):
                if len(method_scores[m]["necessity"]) <= i:
                    method_scores[m]["necessity"].append([])
                    method_scores[m]["sufficiency"].append([])
                method_scores[m]["necessity"][i].append(group[m][k]["necessity"])
                method_scores[m]["sufficiency"][i].append(group[m][k]["sufficiency"])
    if k_vals is None:
        logger.warning("No k values found for plotting.")
        return

    method_colors = {
        "TreeFinder": "lightblue",
        "ContextCite": "lightcoral",
        "TracLLM": "plum",
        "Ground Truth": "lightgreen",
    }

    # Create side-by-side plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

    # Plot necessity scores
    for m in methods:
        means_n = [np.mean(x) for x in method_scores[m]["necessity"]]
        errors_n = [
            np.std(x) / np.sqrt(len(x)) if len(x) > 0 else 0
            for x in method_scores[m]["necessity"]
        ]
        ax1.errorbar(
            k_vals,
            means_n,
            yerr=errors_n,
            marker="o",
            label=f"{m}",
            capsize=3,
            color=method_colors[m],
        )
    ax1.set_xlabel("w (top-w group size)", fontsize=22)
    ax1.set_ylabel("Score", fontsize=22)
    ax1.set_title("Necessity for Top-w Groups", fontsize=26)

    legend_elements = [
        Patch(facecolor=color, alpha=0.7, label=method)
        for method, color in method_colors.items()
    ]
    ax1.legend(handles=legend_elements, fontsize=22, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax1.tick_params(axis="both", which="major", labelsize=18)
    ax1.tick_params(axis="both", which="minor", labelsize=16)
    ax1.set_xticks(k_vals)
    ax1.set_xticklabels([str(k) for k in k_vals], rotation=0)

    # Plot sufficiency scores
    for m in methods:
        means_s = [np.mean(x) for x in method_scores[m]["sufficiency"]]
        errors_s = [
            np.std(x) / np.sqrt(len(x)) if len(x) > 0 else 0
            for x in method_scores[m]["sufficiency"]
        ]
        ax2.errorbar(
            k_vals,
            means_s,
            yerr=errors_s,
            marker="x",
            label=f"{m}",
            capsize=3,
            color=method_colors[m],
        )
    ax2.set_xlabel("w (top-w group size)", fontsize=22)
    ax2.set_title("Sufficiency for Top-w Groups", fontsize=26)
    ax2.legend(handles=legend_elements, fontsize=22, loc="upper right")
    ax2.grid(True, alpha=0.3)

    ax2.tick_params(axis="both", which="major", labelsize=18)
    ax2.tick_params(axis="both", which="minor", labelsize=16)
    ax2.set_xticks(k_vals)
    ax2.set_xticklabels([str(k) for k in k_vals], rotation=0)

    for m in methods:
        combined_scores = [
            (np.mean(n) + np.mean(s)) / 2
            for n, s in zip(
                method_scores[m]["necessity"], method_scores[m]["sufficiency"]
            )
        ]
        errors_combined = [
            np.sqrt((np.std(n) ** 2 + np.std(s) ** 2) / len(n)) if len(n) > 0 else 0
            for n, s in zip(
                method_scores[m]["necessity"], method_scores[m]["sufficiency"]
            )
        ]
        ax3.errorbar(
            k_vals,
            combined_scores,
            yerr=errors_combined,
            marker="s",
            label=f"{m}",
            capsize=3,
            color=method_colors[m],
        )
    ax3.set_xlabel("w (top-w group size)", fontsize=22)
    ax3.set_title("Average for Top-w Groups", fontsize=26)
    ax3.legend(handles=legend_elements, fontsize=22, loc="upper right")
    ax3.grid(True, alpha=0.3)

    ax3.tick_params(axis="both", which="major", labelsize=18)
    ax3.tick_params(axis="both", which="minor", labelsize=16)
    ax3.set_xticks(k_vals)
    ax3.set_xticklabels([str(k) for k in k_vals], rotation=0)

    plt.tight_layout()
    plt.savefig(
        output_dir
        / f"{args['dataset']}_topk_group_scores_{args['version']}_{args['num_ablations']}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    logger.info("Plotted top-w group necessity/sufficiency with outlier control.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive comparison script for TreeFinder vs ContextCite methods"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["hotpot_qa", "loogle_short", "loogle_long", "longbench"],
        default="hotpot_qa",
        help="Dataset to use for comparison",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-1M",
        help="Model name to use",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of samples to process"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for processing"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "TreeFinder",
            "ContextCite",
            "TracLLM",
            "necessity_sufficiency",
            "aggregate_plots",
            "ranking_analysis",
            "all",
            "metrics",
            "topw_group_scores",
            "get_best_sentences",
        ],
        default="all",
        help="Execution mode for the script",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Version identifier for output files",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Alpha value for combined score weighting (0.0 to 1.0)",
    )
    parser.add_argument(
        "--num_ablations",
        type=int,
        default=32,
        help="Number of ablations for context cite mode",
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    torch.set_float32_matmul_precision("high")

    # launch blocking to ensure CUDA is initialized before any other operations
    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Parse command line arguments
    cmd_args = parse_arguments()

    # Configuration
    args = {
        "model_name": cmd_args.model_name,
        "model_path": "",
        "threshold": 0.0,
        "factor": 6,
        "cum_prob": 1.0,
        "topk": 3,
        "batch_size": cmd_args.batch_size,
        "alpha": cmd_args.alpha,
        "dataset": cmd_args.dataset,
        "num_samples": cmd_args.num_samples,
        "output_dir": cmd_args.output_dir,
        "version": cmd_args.version,
        "num_ablations": cmd_args.num_ablations,
    }

    if args["factor"] < 2 or args["factor"] < args["topk"]:
        raise ValueError("Factor must be greater than topk")

    # Setup
    logger = setup_logging()
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting comprehensive comparison...")

    # Handle different execution modes
    if cmd_args.mode == "aggregate_plots":
        run_aggregate_plots_mode(args, logger)
        return
    elif cmd_args.mode == "ranking_analysis":
        run_ranking_analysis_mode(args, logger)
        return

    # For modes that need models and dataset, initialize them
    model, tokenizer = initialize_models(args["model_name"], cmd_args.mode)
    dataset = load_and_prepare_dataset(tokenizer, args["dataset"], args["num_samples"])

    if cmd_args.mode == "TreeFinder":
        run_tree_finder_mode(args, logger, dataset, model, tokenizer)
    elif cmd_args.mode == "ContextCite":
        run_context_cite_mode(args, logger, dataset, model, tokenizer)
    elif cmd_args.mode == "TracLLM":
        run_tracllm_mode(args, logger, dataset, model, tokenizer)
    elif cmd_args.mode == "necessity_sufficiency":
        run_necessity_sufficiency_mode(args, logger, dataset, model, tokenizer)
    elif cmd_args.mode == "topw_group_scores":
        run_topw_group_scores_mode(args, dataset, logger)
    elif cmd_args.mode == "all":
        # Run all modes in sequence
        logger.info("Running all modes in sequence...")

        # 1. TreeFinder
        run_tree_finder_mode(args, logger, dataset, model, tokenizer)

        # 2. ContextCite
        run_context_cite_mode(args, logger, dataset, model, tokenizer)

        # 3. TracLLM
        run_tracllm_mode(args, logger, dataset, model, tokenizer)

        # 4. Necessity/Sufficiency
        run_necessity_sufficiency_mode(args, logger, dataset, model, tokenizer)

        # 5. Aggregate plots
        run_aggregate_plots_mode(args, logger)

        # 6. Ranking analysis
        run_ranking_analysis_mode(args, logger)

        # 7. Top-w group scores
        run_topw_group_scores_mode(args, dataset, logger)

        plot_topw_group_scores(args, logger)

        # 7. Print out the best sentences from each method
        run_get_best_sentences_mode(args, logger, dataset)

        logger.info("All modes completed successfully!")
    elif cmd_args.mode == "metrics":
        run_aggregate_plots_mode(args, logger)

        run_ranking_analysis_mode(args, logger)

        plot_topw_group_scores(args, logger)
    elif cmd_args.mode == "get_best_sentences":
        run_get_best_sentences_mode(args, logger, dataset)

    logger.info("Execution completed.")


if __name__ == "__main__":
    main()
