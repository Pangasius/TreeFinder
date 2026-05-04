#!/usr/bin/env python3
"""Comprehensive comparison script for TreeFinder vs ContextCite vs TracLLM vs SelfCitation."""

import argparse
import json
import logging
import os
import re
import time
import traceback
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from matplotlib.patches import Patch
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from context_cite import ContextCiter
from tracllmkit import PerturbationBasedAttribution
from .tree_finder import TreeFinderTransformer
from .tree_utils import DEFAULT_PROMPT_TEMPLATE, DEFAULT_GENERATE_KWARGS

torch._dynamo.config.suppress_errors = True

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METHODS = ["TreeFinder", "ContextCite", "TracLLM", "SelfCitation", "Ground Truth"]
METHOD_COLORS = {
    "TreeFinder": "lightblue",
    "ContextCite": "lightcoral",
    "TracLLM": "plum",
    "Ground Truth": "lightgreen",
    "GroundTruth": "lightgreen",
    "SelfCitation": "khaki",
}

SELF_CITATION_PROMPT_TEMPLATE = (
    "###Context:\n{context}\n###Question:\n{query}\n\n"
    "Answer the question using ONLY the context above. "
    "After your answer, list the sentences from the context that you used, "
    "ordered by importance (most important first). "
    "Copy each sentence VERBATIM at least up to the tenth word and enclose it in braces like this: "
    "{{sentence}}. Do NOT paraphrase or modify the sentences in any way."
)

LLM_JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator. You will be given a question, a set of "
    "selected sentences from a longer context, and optionally an answer. "
    "Rate on a scale from 1 to 5 how well the selected sentences satisfy the "
    "criterion described by the user.\n"
    "Use exactly this scale:\n"
    "  1 = No\n  2 = No, but (some marginal relevance)\n  3 = Unsure\n"
    "  4 = Yes, but (partially)\n  5 = Yes\n"
    "Reply with ONLY the number (1-5), nothing else."
)


# ---------------------------------------------------------------------------
# Context partitioner (needed by ContextCiter)
# ---------------------------------------------------------------------------
class CustomContextPartitioner:
    """Partitioner that wraps pre-split sentences for ContextCiter."""

    def __init__(self, context: str, sentences: List[str]) -> None:
        self.context, self.sentences = context, sentences

    @property
    def num_sources(self) -> int:
        return len(self.sentences)

    def get_source(self, index: int) -> str:
        return self.sentences[index]

    def get_context(self, mask=None) -> str:
        if mask is None:
            return self.context
        return "".join(s for s, m in zip(self.sentences, mask) if m)

    @property
    def sources(self) -> List[str]:
        return list(self.sentences)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def convert_tensors(obj):
    """Recursively convert tensors / ndarrays to plain Python for JSON."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "cpu"):
        return obj.cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {k: convert_tensors(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_tensors(v) for v in obj]
    return obj


def save_results(results, path, log):
    with open(path, "w") as f:
        json.dump(convert_tensors(results), f, indent=2)
    log.info(f"Saved {len(results)} results to {path}")


def load_results(path, log):
    """Load existing results → (list, set-of-processed-indices)."""
    if path.exists():
        try:
            with open(path) as f:
                res = json.load(f)
            idx_set = {r["data_index"] for r in res if "data_index" in r}
            log.info(f"Loaded {len(res)} results from {path}")
            return res, idx_set
        except Exception as e:
            log.warning(f"Failed to load {path}: {e}")
    return [], set()


def compute_combined(nec, suf, alpha=0.5):
    return [alpha * n + (1 - alpha) * s for n, s in zip(nec, suf)]


def rank_indices(scores, reverse=True):
    """Return sentence indices sorted by *scores*."""
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=reverse)


def _result_path(output_dir, dataset, method, version, num_ablations=None):
    """Centralise output-file naming."""
    d = Path(output_dir)
    if method == "ContextCite":
        return d / f"{dataset}_ContextCite_{num_ablations}.json"
    if method == "necessity_sufficiency":
        return d / f"{dataset}_necessity_sufficiency.json"
    if method == "TracLLM":
        return d / f"{dataset}_TracLLM.json"
    if method == "SelfCitation":
        return d / f"{dataset}_SelfCitation.json"
    return d / f"{dataset}_{method}_{version}.json"


def _load_method_results(
    args,
    log,
    methods=(
        "TreeFinder",
        "ContextCite",
        "TracLLM",
        "SelfCitation",
        "necessity_sufficiency",
    ),
):
    """Load result files for *methods* and return {name: (results, idx_set)}."""
    out = {}
    for m in methods:
        p = _result_path(
            args["output_dir"],
            args["dataset"],
            m,
            args["version"],
            args.get("num_ablations"),
        )
        out[m] = load_results(p, log)
    return out


def _find(results, data_idx):
    """Find a result dict by data_index (linear scan — lists are small)."""
    return next((r for r in results if r["data_index"] == data_idx), None)


# ---------------------------------------------------------------------------
# Dataset & model initialisation
# ---------------------------------------------------------------------------
def load_and_prepare_dataset(tokenizer, dataset_name="hotpot_qa", num_samples=100):
    count = lambda t: len(tokenizer.encode(t, add_special_tokens=False))
    loaders = {
        "hotpot_qa": lambda: load_dataset(
            "hotpotqa/hotpot_qa", "fullwiki", split="train", trust_remote_code=True
        ),
        "loogle_short": lambda: load_dataset(
            "bigai-nlco/LooGLE", "shortdep_qa", split="test"
        ),
        "loogle_long": lambda: load_dataset(
            "bigai-nlco/LooGLE", "longdep_qa", split="test"
        ),
        "longbench": lambda: load_dataset("THUDM/LongBench-v2", split="train"),
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    ds = loaders[dataset_name]()
    if dataset_name != "hotpot_qa":
        ds = ds.filter(lambda x: count(x["context"]) < 20000)
    print(f"Dataset {dataset_name} loaded with {len(ds)} samples.")
    ds = ds.select(range(min(num_samples, len(ds))))
    logger.info(f"Loaded {len(ds)} samples")
    return ds


def preprocess_sample(sample, dataset="hotpot_qa"):
    q, ctx, ans = sample["question"], sample["context"], sample["answer"]
    if dataset == "hotpot_qa":
        sents = [s for title, s in ctx.items() if title != "title"]
        sents = [s for sub in sents for s in sub]
        sents = [s for sub in sents for s in sub]
        ctx = " ".join(sents)
    return {
        "question": q,
        "context": ctx,
        "sentences": sent_tokenize(ctx),
        "answer": ans,
    }


def initialize_models(model_name="Qwen/Qwen2.5-1.5B-Instruct", mode="default"):
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if mode == "metrics":
        return None, tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    model.eval()
    model.generation_config.cache_implementation = "dynamic"
    return model, tokenizer


def _make_finder(args, model, tokenizer, log):
    return TreeFinderTransformer(
        model_name=args["model_name"],
        model_path=args["model_path"],
        model=model,
        tokenizer=tokenizer,
        factor=args["factor"],
        topk=args["topk"],
        batch_size=args["batch_size"],
        logger=log,
        alpha=args["alpha"],
        expansion_budget=args["num_ablations"],
    )


# ---------------------------------------------------------------------------
# Generic processing loop
# ---------------------------------------------------------------------------
def _process_loop(dataset, args, log, output_path, desc, process_fn):
    """Iterate over *dataset*, call *process_fn(idx, processed_sample)*, save."""
    results, done = load_results(output_path, log)
    for idx in tqdm(range(len(dataset)), desc=desc):
        if idx in done:
            continue
        try:
            sample = preprocess_sample(dataset[idx], args["dataset"])
            result = process_fn(idx, sample)
            if result is not None:
                results.append(result)
            if len(results) % 10 == 0 or idx == len(dataset) - 1:
                save_results(results, output_path, log)
        except Exception as e:
            log.error(f"Error sample {idx}: {e}")
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Mode: TreeFinder
# ---------------------------------------------------------------------------
def run_tree_finder_mode(args, log, dataset, model, tokenizer):
    finder = _make_finder(args, model, tokenizer, log)
    path = _result_path(
        args["output_dir"], args["dataset"], "TreeFinder", args["version"]
    )

    def process(idx, s):
        finder.get_and_reset_num_calls()
        t0 = time.time()
        scores, answer = finder.get_scores(s["question"], s["sentences"])
        return {
            "data_index": idx,
            "scores": scores.tolist(),
            "num_calls": finder.get_and_reset_num_calls(),
            "time": time.time() - t0,
            "answer": answer,
        }

    return _process_loop(dataset, args, log, path, "TreeFinder", process)


# ---------------------------------------------------------------------------
# Mode: ContextCite
# ---------------------------------------------------------------------------
def run_context_cite_mode(args, log, dataset, model, tokenizer):
    path = _result_path(
        args["output_dir"],
        args["dataset"],
        "ContextCite",
        args["version"],
        args["num_ablations"],
    )

    def process(idx, s):
        part = CustomContextPartitioner(s["context"], s["sentences"])
        cc = ContextCiter(
            model=model,
            tokenizer=tokenizer,
            context=s["context"],
            query=s["question"],
            prompt_template=DEFAULT_PROMPT_TEMPLATE,
            generate_kwargs=DEFAULT_GENERATE_KWARGS,
            num_ablations=args["num_ablations"],
            batch_size=args["batch_size"],
            partitioner=part,
        )
        t0 = time.time()
        w = cc.get_attributions(start_idx=0, end_idx=len(cc.response), verbose=False)
        return {
            "data_index": idx,
            "scores": w.tolist(),
            "num_calls": args["num_ablations"],
            "time": time.time() - t0,
            "answer": cc.response,
        }

    return _process_loop(dataset, args, log, path, "ContextCite", process)


# ---------------------------------------------------------------------------
# Mode: TracLLM
# ---------------------------------------------------------------------------
def run_tracllm_mode(args, log, dataset, model, tokenizer):
    path = _result_path(args["output_dir"], args["dataset"], "TracLLM", args["version"])

    @dataclass
    class _LLMWrap:
        model: Any
        tokenizer: Any

        @property
        def name(self):
            return args["model_name"]

    def process(idx, s):
        trac = PerturbationBasedAttribution(
            llm=_LLMWrap(model, tokenizer),
            explanation_level="sentence",
            attr_type="tracllm",
            K=args["topk"],
        )
        prompt = DEFAULT_PROMPT_TEMPLATE.format(
            query=s["question"], context=" ".join(s["sentences"])
        )
        tok = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        )
        t0 = time.time()
        out = model.generate(
            inputs=tok.input_ids.to(model.device),
            return_dict_in_generate=True,
            output_scores=False,
            attention_mask=tok.attention_mask.to(model.device),
            **DEFAULT_GENERATE_KWARGS,
        )
        answer = tokenizer.decode(
            out.sequences[0, tok.input_ids.shape[1]:], skip_special_tokens=True
        )
        _, ids, scores, _, _ = trac.attribute(s["question"], s["sentences"], answer)
        all_s = [-float("inf")] * len(s["sentences"])
        for i, sc in zip(ids, scores):
            all_s[i] = sc
        res = {
            "data_index": idx,
            "scores": all_s,
            "num_calls": trac.num_calls,
            "time": time.time() - t0,
            "answer": answer,
        }
        trac.num_calls = 0
        return res

    return _process_loop(dataset, args, log, path, "TracLLM", process)


# ---------------------------------------------------------------------------
# Mode: SelfCitation
# ---------------------------------------------------------------------------
def _parse_self_citations(response):
    return [c.strip() for c in re.findall(r"\{([^}]+)\}", response) if c.strip()]


def _match_citations(citations, sentences, threshold=0.3):
    matched, used = [], set()
    for cit in citations:
        best_idx, best_r = -1, 0.0
        # Get first ~10 words from citation for matching
        cit_words = cit.lower().split()[:10]
        cit_prefix = " ".join(cit_words)

        for i, s in enumerate(sentences):
            if i in used:
                continue
            # Get first ~10 words from sentence for comparison
            s_words = s.lower().split()[:10]
            s_prefix = " ".join(s_words)

            r = SequenceMatcher(None, cit_prefix, s_prefix).ratio()
            if r > best_r:
                best_r, best_idx = r, i
        if best_idx >= 0 and best_r >= threshold:
            matched.append(best_idx)
            used.add(best_idx)
    return matched


def run_self_citation_mode(args, log, dataset, model, tokenizer):
    path = _result_path(
        args["output_dir"], args["dataset"], "SelfCitation", args["version"]
    )

    def process(idx, s):
        prompt = SELF_CITATION_PROMPT_TEMPLATE.format(
            context=s["context"], query=s["question"]
        )
        tok = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=False,
            return_tensors="pt",
        )
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                inputs=tok.input_ids.to(model.device),
                attention_mask=tok.attention_mask.to(model.device),
                max_new_tokens=1024,
                do_sample=False,
                temperature=None,
                top_p=None,
                repetition_penalty=1.0,
            )
        resp = tokenizer.decode(out[0, tok.input_ids.shape[1]:], skip_special_tokens=True)
        elapsed = time.time() - t0
        cits = _parse_self_citations(resp)
        matched = _match_citations(cits, s["sentences"])
        n = len(matched)
        scores = [-float("inf")] * len(s["sentences"])
        for rank, si in enumerate(matched):
            scores[si] = float(n - rank) # Higher score for higher-ranked citations (1st cited = n, 2nd = n-1, etc.)
        return {
            "data_index": idx,
            "scores": scores,
            "num_calls": 1,
            "time": elapsed,
            "num_citations_parsed": len(cits),
            "num_citations_matched": n,
            "answer": resp,
        }

    return _process_loop(dataset, args, log, path, "SelfCitation", process)


# ---------------------------------------------------------------------------
# Mode: Necessity / Sufficiency
# ---------------------------------------------------------------------------
def run_necessity_sufficiency_mode(args, log, dataset, model, tokenizer):
    finder = _make_finder(args, model, tokenizer, log)
    path = _result_path(
        args["output_dir"], args["dataset"], "necessity_sufficiency", args["version"]
    )

    def process(idx, s):
        nec, suf = finder.get_necessity_sufficiency(
            s["question"],
            s["sentences"],
            indices=[[i] for i in range(len(s["sentences"]))],
        )
        return {
            "data_index": idx,
            "necessity_scores": nec.tolist(),
            "sufficiency_scores": suf.tolist(),
        }

    return _process_loop(dataset, args, log, path, "Necessity/Sufficiency", process)


# ---------------------------------------------------------------------------
# Mode: Aggregate plots
# ---------------------------------------------------------------------------
def run_aggregate_plots_mode(args, log):
    output_dir = Path(args["output_dir"])
    data = _load_method_results(
        args, log, ("TreeFinder", "ContextCite", "TracLLM", "necessity_sufficiency", "SelfCitation")
    )
    tf_res, cc_res = data["TreeFinder"][0], data["ContextCite"][0]
    tl_res, ns_res = data["TracLLM"][0], data["necessity_sufficiency"][0]
    sc_res = data["SelfCitation"][0]

    combined_results = []
    for ns in ns_res:
        di = ns["data_index"]
        tf, cc, tl, sc = _find(tf_res, di), _find(cc_res, di), _find(tl_res, di), _find(sc_res, di)
        if not (tf and cc and tl and sc):
            continue
        comb = compute_combined(ns["necessity_scores"], ns["sufficiency_scores"])
        rankings = {
            "TreeFinder": rank_indices(tf["scores"], reverse=False)[:10],
            "ContextCite": rank_indices(cc["scores"])[:10],
            "TracLLM": rank_indices(tl["scores"])[:10],
            "SelfCitation": rank_indices(sc["scores"])[:10],
        }
        gt_top10 = rank_indices(comb, reverse=False)[:10]
        cr = {"data_index": di, "num_sentences": len(comb)}
        for method, top in rankings.items():
            cr[f"{method}_necessity_scores"] = [ns["necessity_scores"][i] for i in top]
            cr[f"{method}_sufficiency_scores"] = [
                ns["sufficiency_scores"][i] for i in top
            ]
            cr[f"{method}_combined_scores"] = [comb[i] for i in top]
            cr[f"{method}_time"] = (
                tf
                if method == "TreeFinder"
                else cc
                if method == "ContextCite"
                else tl
                if method == "TracLLM"
                else sc
            )["time"]
            cr[f"{method}_num_calls"] = (
                tf
                if method == "TreeFinder"
                else cc
                if method == "ContextCite"
                else tl
                if method == "TracLLM"
                else sc
            )["num_calls"]
        for key in ("necessity_scores", "sufficiency_scores", "combined_scores"):
            src = ns if key != "combined_scores" else {"combined_scores": comb}
            cr[key] = [src[key][i] for i in gt_top10]
        combined_results.append(cr)

    if combined_results:
        _create_aggregate_plots(
            combined_results,
            output_dir,
            f"{args['dataset']}_top10_{args['version']}_{args['num_ablations']}",
        )
    else:
        log.warning("No matching results for aggregate plots")


def _create_aggregate_plots(results, output_dir, dataset):
    valid = [r for r in results if r is not None]
    avgs = {
        m: {
            "time": np.mean([r[f"{m}_time"] for r in valid]),
            "calls": np.mean([r[f"{m}_num_calls"] for r in valid]),
        }
        for m in ("TreeFinder", "ContextCite", "TracLLM", "SelfCitation")
    }
    max_len = max(len(r["necessity_scores"]) for r in valid)
    metrics = ["necessity", "sufficiency", "combined"]
    method_keys = {
        "TreeFinder": "TreeFinder",
        "ContextCite": "ContextCite",
        "TracLLM": "TracLLM",
        "SelfCitation": "SelfCitation",
        "Ground Truth": "",
    }
    all_data = {
        mk: {m: [[] for _ in range(max_len)] for m in metrics} for mk in method_keys
    }
    for r in valid:
        for m in metrics:
            for mk, prefix in method_keys.items():
                key = f"{prefix}_{m}_scores" if prefix else f"{m}_scores"
                vals = r.get(key, [])
                for i, v in enumerate(vals):
                    all_data[mk][m][i].append(v)

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    colors_list = ["lightblue", "lightcoral", "plum", "khaki", "lightgreen"]
    method_names = list(method_keys)

    for ax_i, metric in enumerate(metrics):
        box_data, colors, positions = [], [], []
        pos = 1
        for i in range(max_len):
            if all_data[method_names[0]][metric][i]:
                for j, mk in enumerate(method_names):
                    vals = np.array(all_data[mk][metric][i])
                    box_data.append(vals[~np.isnan(vals)])
                    colors.append(colors_list[j])
                    positions.append(pos + j)
                pos += len(method_names) + 1

        bp = axes[ax_i].boxplot(
            box_data,
            positions=positions,
            patch_artist=True,
            showfliers=True,
            widths=0.8,
            whis=[1, 99],
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        for part in ("whiskers", "caps"):
            for el in bp[part]:
                el.set(color="gray", alpha=0.5, linewidth=1.0)
        for fl in bp["fliers"]:
            fl.set(
                marker="o",
                markerfacecolor="gray",
                markeredgecolor="gray",
                alpha=0.3,
                markersize=3,
            )

        legend_el = [
            Patch(facecolor=c, alpha=0.7, label=n)
            for n, c in zip(method_names, colors_list)
        ]
        axes[ax_i].set_title(f"{metric.capitalize()} Scores", fontsize=26)
        if ax_i == 1:
            axes[ax_i].set_xlabel("Sentence Index (Ordered by Score)", fontsize=22)
        if ax_i == 0:
            axes[ax_i].set_ylabel("Score", fontsize=22)
        axes[ax_i].legend(handles=legend_el, fontsize=18)
        axes[ax_i].grid(True, alpha=0.3)
        nm = len(method_names)
        centers = [np.mean(positions[i: i + nm]) for i in range(0, len(positions), nm)]
        axes[ax_i].set_xticks(centers)
        axes[ax_i].set_xticklabels([str(i + 1) for i in range(len(centers))])
        axes[ax_i].tick_params(axis="both", which="major", labelsize=18)

    timing = " | ".join(
        f"{m}: {avgs[m]['time']:.2f}s ({avgs[m]['calls']:.0f} calls)" for m in avgs
    )
    fig.text(
        0.5,
        0.02,
        timing,
        ha="center",
        fontsize=20,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7),
    )
    plt.tight_layout(pad=4.0, h_pad=4.0)
    plt.savefig(
        output_dir / f"{dataset}_aggregate_comparison_three_metrics.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mode: Ranking analysis
# ---------------------------------------------------------------------------
def run_ranking_analysis_mode(args, log):
    output_dir = Path(args["output_dir"])
    data = _load_method_results(
        args, log, ("TreeFinder", "ContextCite", "TracLLM", "necessity_sufficiency", "SelfCitation")
    )
    tf_res, cc_res = data["TreeFinder"][0], data["ContextCite"][0]
    tl_res, ns_res = data["TracLLM"][0], data["necessity_sufficiency"][0]
    sc_res = data["SelfCitation"][0]

    positions_by_k = {
        m: {k: [] for k in range(1, 6)}
        for m in ("TreeFinder", "ContextCite", "TracLLM", "SelfCitation")
    }

    for ns in ns_res:
        di = ns["data_index"]
        tf, cc, tl, sc = _find(tf_res, di), _find(cc_res, di), _find(tl_res, di), _find(sc_res, di)
        if not (tf and cc and tl and sc):
            continue
        comb = compute_combined(
            ns["necessity_scores"], ns["sufficiency_scores"], alpha=args["alpha"]
        )
        gt_rank = rank_indices(comb, reverse=False)
        rankings = {
            "TreeFinder": rank_indices(tf["scores"], reverse=False),
            "ContextCite": rank_indices(cc["scores"]),
            "TracLLM": rank_indices(tl["scores"]),
            "SelfCitation": rank_indices(sc["scores"]),
        }
        for m, rk in rankings.items():
            for k in range(1, 6):
                if len(rk) >= k:
                    positions_by_k[m][k].extend(gt_rank.index(si) + 1 for si in rk[:k])

    if not any(positions_by_k[m][k] for m in positions_by_k for k in range(1, 6)):
        log.warning("No valid ranking data")
        return

    ranking_results = {}
    for k in range(1, 6):
        for m in positions_by_k:
            if positions_by_k[m][k]:
                med = np.median(positions_by_k[m][k])
                log.info(f"k={k} {m}: median GT position = {med:.2f}")
                ranking_results[f"k{k}_{m}_median_gt_position"] = med
                ranking_results[f"k{k}_{m}_positions"] = positions_by_k[m][k]

    rpath = output_dir / f"{args['dataset']}_ranking_analysis_{args['version']}.json"
    with open(rpath, "w") as f:
        json.dump(ranking_results, f, indent=2)
    log.info(f"Ranking analysis saved to {rpath}")


# ---------------------------------------------------------------------------
# Mode: Top-w group scores
# ---------------------------------------------------------------------------
def run_topw_group_scores_mode(args, dataset, log):
    output_dir = Path(args["output_dir"])
    k_max = 5
    data = _load_method_results(args, log)
    tf_res, cc_res = data["TreeFinder"][0], data["ContextCite"][0]
    tl_res, sc_res = data["TracLLM"][0], data["SelfCitation"][0]
    ns_res = data["necessity_sufficiency"][0]

    output_path = (
        output_dir
        / f"{args['dataset']}_topk_group_scores_{args['version']}_{args['num_ablations']}.json"
    )
    all_gs, done = load_results(output_path, log)

    model, tokenizer = initialize_models(args["model_name"], "default")
    finder = _make_finder(args, model, tokenizer, log)

    for si in tqdm(range(len(ns_res)), desc="Top-w groups"):
        ns = ns_res[si]
        di = ns["data_index"]
        if di in done:
            if si < len(all_gs) and all(m in all_gs[si] for m in METHODS):
                continue
        tf, cc, tl, sc = (
            _find(tf_res, di),
            _find(cc_res, di),
            _find(tl_res, di),
            _find(sc_res, di),
        )
        if not all((tf, cc, tl, sc)):
            log.warning(f"Missing results for index {di}, skipping.")
            continue

        s = preprocess_sample(dataset[di], args["dataset"])
        sents, q, n = s["sentences"], s["question"], len(s["sentences"])
        gt_comb = compute_combined(ns["necessity_scores"], ns["sufficiency_scores"])
        rankings = {
            "TreeFinder": rank_indices(tf["scores"], reverse=False),
            "ContextCite": rank_indices(cc["scores"]),
            "TracLLM": rank_indices(tl["scores"]),
            "SelfCitation": rank_indices(sc["scores"]),
            "Ground Truth": rank_indices(gt_comb, reverse=False),
        }
        gs = {"data_index": di}
        for method, rk in rankings.items():
            gs[method] = {}
            masks = [rk[:k] for k in range(2, min(k_max, n) + 1)]
            nec, suf = finder.get_necessity_sufficiency(q, sents, masks)
            for k in range(2, min(k_max, n) + 1):
                gs[method][f"k={k}"] = {
                    "necessity": float(nec[k - 2]),
                    "sufficiency": float(suf[k - 2]),
                }
            gs[method]["k=1"] = {
                "necessity": float(ns["necessity_scores"][rk[0]]),
                "sufficiency": float(ns["sufficiency_scores"][rk[0]]),
            }
        all_gs.append(gs)
        if len(all_gs) % 10 == 0 or si == len(ns_res) - 1:
            save_results(all_gs, output_path, log)

    save_results(all_gs, output_path, log)


# ---------------------------------------------------------------------------
# Mode: Get best sentences
# ---------------------------------------------------------------------------
def run_get_best_sentences_mode(args, log, dataset):
    output_dir = Path(args["output_dir"])
    data = _load_method_results(args, log)
    tf_res, cc_res = data["TreeFinder"][0], data["ContextCite"][0]
    tl_res, sc_res = data["TracLLM"][0], data["SelfCitation"][0]
    ns_res = data["necessity_sufficiency"][0]

    out = (
        output_dir
        / f"{args['dataset']}_best_sentences_{args['version']}_{args['num_ablations']}.txt"
    )

    with open(out, "w") as f:
        for ns in ns_res:
            di = ns["data_index"]
            tf, cc, tl, sc = (
                _find(tf_res, di),
                _find(cc_res, di),
                _find(tl_res, di),
                _find(sc_res, di),
            )
            if not all((tf, cc, tl, sc)):
                continue
            s = preprocess_sample(dataset[di], args["dataset"])
            sents, q, n = s["sentences"], s["question"], len(s["sentences"])
            gt_comb = compute_combined(ns["necessity_scores"], ns["sufficiency_scores"])
            all_scores = {
                "TreeFinder": tf["scores"],
                "ContextCite": cc["scores"],
                "TracLLM": tl["scores"],
                "SelfCitation": sc["scores"],
                "Ground Truth": gt_comb,
            }
            rankings = {
                m: (rank_indices(v, reverse=(m != "Ground Truth" and m != "TreeFinder")))
                for m, v in all_scores.items()
            }

            f.write(f"Data Index: {di}\nQuestion: {q}\n\n")
            for method, rk in rankings.items():
                f.write(f"Top 5 sentences from {method}:\n")
                for rank in range(min(5, len(rk))):
                    si = rk[rank]
                    f.write(
                        f"Rank {rank + 1} (Score: {all_scores[method][si]:.4g}): "
                        f"{sents[si]}\n"
                    )
                f.write("\n")
            f.write("=" * 80 + "\n\n")
    log.info(f"Saved best sentences to {out}")


# ---------------------------------------------------------------------------
# Mode: LLM-as-a-Judge
# ---------------------------------------------------------------------------
def _generate(model, tokenizer, messages, **gen_kw):
    """Tokenise, generate, decode — returns the generated text only."""
    tokenized_messages = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=True, return_tensors="pt",
    )
    with torch.no_grad():
        out = model.generate(inputs=tokenized_messages.input_ids.to(model.device), **gen_kw, attention_mask=tokenized_messages.attention_mask.to(model.device))
    return tokenizer.decode(
        out[0, tokenized_messages.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


def llm_judge_evaluate(model, tokenizer, question, selected, gt_answer, model_answer):
    criteria = {
        "answers_question": (
            "Do the selected sentences contain enough information to answer the question?",
            None,
        ),
        "explains_ground_truth": (
            "Do the selected sentences lead one to answer with the following answer?",
            gt_answer,
        ),
        "explains_model_answer": (
            "Do the selected sentences lead one to answer with the following answer?",
            model_answer,
        ),
    }
    scores = {}
    for key, (crit, ans) in criteria.items():
        sents_txt = "\n".join(f"  [{i + 1}] {s}" for i, s in enumerate(selected))
        prompt = f"Question: {question}\n\nSelected sentences:\n{sents_txt}\n\n"
        if ans:
            prompt += f"Answer: {ans}\n\n"
        prompt += f"Criterion: {crit}\nYour rating (1-5):"
        resp = _generate(
            model,
            tokenizer,
            [
                {"role": "system", "content": LLM_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=4,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

        match = re.search(r'\d+', resp)
        if match:
            scores[key] = max(1, min(5, int(match.group())))
        else:
            scores[key] = 3

    return scores


def run_llm_judge_mode(args, log, dataset, model, tokenizer, top_x=5):
    output_dir = Path(args["output_dir"])
    data = _load_method_results(
        args, log, ("TreeFinder", "ContextCite", "TracLLM", "necessity_sufficiency", "SelfCitation")
    )
    tf_res, cc_res = data["TreeFinder"][0], data["ContextCite"][0]
    tl_res, ns_res = data["TracLLM"][0], data["necessity_sufficiency"][0]
    sc_res = data["SelfCitation"][0]

    out_path = (
        output_dir
        / f"{args['dataset']}_llm_judge_{args['version']}_{args['num_ablations']}.json"
    )
    judge_res, done = load_results(out_path, log)

    for ns in tqdm(ns_res, desc="LLM Judge"):
        di = ns["data_index"]
        if di in done:
            continue
        tf, cc, tl, sc = _find(tf_res, di), _find(cc_res, di), _find(tl_res, di), _find(sc_res, di)
        if not (tf and cc and tl and sc):
            continue
        try:
            s = preprocess_sample(dataset[di], args["dataset"])
            q, sents, gt = s["question"], s["sentences"], s["answer"]

            rankings = {
                "TreeFinder": rank_indices(tf["scores"], reverse=False)[:top_x],
                "ContextCite": rank_indices(cc["scores"])[:top_x],
            }

            tl_valid = [i for i, v in enumerate(tl["scores"]) if v != -float("inf")]
            rankings["TracLLM"] = sorted(
                tl_valid, key=lambda i: tl["scores"][i], reverse=True
            )[:top_x]

            sc_valid = [i for i, v in enumerate(sc["scores"]) if v != -float("inf")]
            rankings["SelfCitation"] = sorted(
                sc_valid, key=lambda i: sc["scores"][i], reverse=True
            )[:top_x]

            comb = compute_combined(ns["necessity_scores"], ns["sufficiency_scores"])
            rankings["GroundTruth"] = rank_indices(comb, reverse=False)[:top_x]

            # Use each method's own generated answer for explainability scoring.
            method_answers = {
                "TreeFinder": tf.get("answer"),
                "ContextCite": cc.get("answer"),
                "TracLLM": tl.get("answer"),
                "SelfCitation": sc.get("answer"),
                "GroundTruth": gt,
            }

            entry = {
                "data_index": di,
                "question": q,
                "ground_truth_answer": gt,
                "model_answers": method_answers,
            }
            for m, rk in rankings.items():
                if not rk:
                    entry[m] = None
                    continue
                entry[m] = llm_judge_evaluate(
                    model,
                    tokenizer,
                    q,
                    [sents[i] for i in rk],
                    gt,
                    method_answers.get(m),
                )
            judge_res.append(entry)
            if len(judge_res) % 10 == 0 or di == ns_res[-1]["data_index"]:
                save_results(judge_res, out_path, log)
        except Exception as e:
            log.error(f"LLM judge error sample {di}: {e}")
            traceback.print_exc()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_results(judge_res, out_path, log)


# ---------------------------------------------------------------------------
# Plotting: LLM judge
# ---------------------------------------------------------------------------
def plot_llm_judge_results(args, log, datasets=None):
    output_dir = Path(args["output_dir"])
    datasets = datasets or [args["dataset"]]
    crit_keys = ["answers_question", "explains_ground_truth", "explains_model_answer"]
    crit_labels = [
        "Attribution enables\nanswering the question",
        "Attribution leads to\nthe ground truth",
        "Attribution leads to\nthe model's answer",
    ]
    methods = ["TreeFinder", "ContextCite", "TracLLM", "GroundTruth", "SelfCitation"]
    scale = ["1\n(Weakly)", "2", "3", "4", "5\n(Strongly)"]
    agg = {m: {c: [] for c in crit_keys} for m in methods}

    for ds in datasets:
        jp = (
            output_dir
            / f"{ds}_llm_judge_{args['version']}_{args['num_ablations']}.json"
        )
        if not jp.exists():
            log.warning(f"No judge results for {ds}")
            continue
        with open(jp) as f:
            jr = json.load(f)
        sc = {m: {c: [] for c in crit_keys} for m in methods}
        for e in jr:
            for m in methods:
                if e.get(m) is None:
                    continue
                for c in crit_keys:
                    sc[m][c].append(e[m][c])
        _plot_judge_chart(
            sc,
            methods,
            crit_keys,
            crit_labels,
            METHOD_COLORS,
            scale,
            f"LLM-as-a-Judge — {ds}",
            output_dir
            / f"{ds}_llm_judge_{args['version']}_{args['num_ablations']}.pdf",
        )
        for m in methods:
            for c in crit_keys:
                if sc[m][c]:
                    agg[m][c].extend(sc[m][c])
                    
    if len(datasets) > 1:
        if any(agg[m][c] for m in methods for c in crit_keys):
            _plot_judge_chart(
                agg,
                methods,
                crit_keys,
                crit_labels,
                METHOD_COLORS,
                scale,
                "LLM-as-a-Judge — Aggregate",
                output_dir
                / f"aggregate_llm_judge_{args['version']}_{args['num_ablations']}.pdf",
            )


def _plot_judge_chart(
    scores, methods, crit_keys, crit_labels, colors, scale, title, save_path
):
    x = np.arange(len(crit_keys))
    bw = 0.18
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, m in enumerate(methods):
        means = [np.mean(scores[m][c]) if scores[m][c] else 0 for c in crit_keys]
        sems = [
            np.std(scores[m][c]) / np.sqrt(len(scores[m][c]))
            if len(scores[m][c]) > 1  # Only compute SEM with 2+ samples
            else np.nan  # or 0, depending on desired behavior
            for c in crit_keys
        ]
        ax.bar(
            x + (i - len(methods) / 2 + 0.5) * bw,
            means,
            bw,
            yerr=sems,
            label=m,
            color=colors.get(m, "gray"),
            edgecolor="gray",
            capsize=3,
            alpha=0.85,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(crit_labels, fontsize=14)
    ax.set_ylabel("Mean Score", fontsize=16)
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(scale, fontsize=12)
    ax.set_title(title, fontsize=20)
    ax.legend(fontsize=13, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting: Top-w group scores
# ---------------------------------------------------------------------------
def plot_topw_group_scores(args, log):
    output_dir = Path(args["output_dir"])
    jp = (
        output_dir
        / f"{args['dataset']}_topk_group_scores_{args['version']}_{args['num_ablations']}.json"
    )
    with open(jp) as f:
        all_gs = json.load(f)

    methods = ["TreeFinder", "ContextCite", "TracLLM", "Ground Truth", "SelfCitation"]
    k_vals, ms = None, {m: {"necessity": [], "sufficiency": []} for m in methods}
    for g in all_gs:
        for m in methods:
            keys = sorted(g[m], key=lambda x: int(x.split("=")[1]))
            if k_vals is None:
                k_vals = [int(x.split("=")[1]) for x in keys]
            for i, k in enumerate(keys):
                if len(ms[m]["necessity"]) <= i:
                    ms[m]["necessity"].append([])
                    ms[m]["sufficiency"].append([])
                ms[m]["necessity"][i].append(g[m][k]["necessity"])
                ms[m]["sufficiency"][i].append(g[m][k]["sufficiency"])
    if not k_vals:
        log.warning("No k values")
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    markers = ["o", "x", "s"]
    titles = [
        "Necessity for Top-w Groups",
        "Sufficiency for Top-w Groups",
        "Average for Top-w Groups",
    ]
    legend_el = [
        Patch(facecolor=METHOD_COLORS.get(m, "gray"), alpha=0.7, label=m)
        for m in methods
    ]

    for ax_i, (metric_key, title) in enumerate(
        zip(["necessity", "sufficiency", None], titles)
    ):
        for m in methods:
            if metric_key:
                vals = [np.mean(x) for x in ms[m][metric_key]]
                errs = [
                    np.std(x) / np.sqrt(len(x)) if x else 0 for x in ms[m][metric_key]
                ]
            else:
                vals = [
                    (np.mean(n) + np.mean(s)) / 2
                    for n, s in zip(ms[m]["necessity"], ms[m]["sufficiency"])
                ]
                errs = [
                    np.sqrt((np.std(n) ** 2 + np.std(s) ** 2) / len(n)) if n else 0
                    for n, s in zip(ms[m]["necessity"], ms[m]["sufficiency"])
                ]
            axes[ax_i].errorbar(
                k_vals,
                vals,
                yerr=errs,
                marker=markers[ax_i],
                capsize=3,
                color=METHOD_COLORS.get(m, "gray"),
                label=m,
            )
        axes[ax_i].set_xlabel("w (top-w group size)", fontsize=22)
        axes[ax_i].set_title(title, fontsize=26)
        axes[ax_i].legend(handles=legend_el, fontsize=18)
        axes[ax_i].grid(True, alpha=0.3)
        axes[ax_i].tick_params(axis="both", which="major", labelsize=18)
        axes[ax_i].set_xticks(k_vals)
        if ax_i == 0:
            axes[ax_i].set_ylabel("Score", fontsize=22)

    plt.tight_layout()
    plt.savefig(
        output_dir
        / f"{args['dataset']}_topk_group_scores_{args['version']}_{args['num_ablations']}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    log.info("Plotted top-w group scores.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
MODES = [
    "TreeFinder",
    "ContextCite",
    "TracLLM",
    "SelfCitation",
    "necessity_sufficiency",
    "aggregate_plots",
    "ranking_analysis",
    "all",
    "metrics",
    "topw_group_scores",
    "get_best_sentences",
    "llm_judge",
    "plot_llm_judge",
]


def parse_arguments():
    p = argparse.ArgumentParser(description="TreeFinder vs ContextCite comparison")
    p.add_argument(
        "--dataset",
        choices=["hotpot_qa", "loogle_short", "loogle_long", "longbench"],
        default="hotpot_qa",
    )
    p.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct-1M")
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--mode", choices=MODES, default="all")
    p.add_argument("--top_x", type=int, default=5)
    p.add_argument("--version", default="v9")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--num_ablations", type=int, default=32)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--factor", type=int, default=5)
    return p.parse_args()


def main():
    torch.set_float32_matmul_precision("high")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    cmd = parse_arguments()
    args = {
        "model_name": cmd.model_name,
        "model_path": "",
        "factor": cmd.factor,
        "topk": cmd.top_k,
        "batch_size": cmd.batch_size,
        "alpha": cmd.alpha,
        "dataset": cmd.dataset,
        "num_samples": cmd.num_samples,
        "output_dir": cmd.output_dir,
        "version": cmd.version,
        "num_ablations": cmd.num_ablations,
        "top_x": cmd.top_x,
    }
    if args["factor"] < 2 or args["factor"] < args["topk"]:
        raise ValueError("Factor must be >= 2 and >= topk")

    log = setup_logging()
    Path(args["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Modes that don't need model/dataset
    no_model_modes = {
        "aggregate_plots": lambda: run_aggregate_plots_mode(args, log),
        "ranking_analysis": lambda: run_ranking_analysis_mode(args, log),
        "plot_llm_judge": lambda: plot_llm_judge_results(args, log),
        "metrics": lambda: (
            run_aggregate_plots_mode(args, log),
            run_ranking_analysis_mode(args, log),
            plot_topw_group_scores(args, log),
            plot_llm_judge_results(args, log),
        ),
    }
    if cmd.mode in no_model_modes:
        no_model_modes[cmd.mode]()
        log.info("Execution completed.")
        return

    model, tokenizer = initialize_models(args["model_name"], cmd.mode)
    ds = load_and_prepare_dataset(tokenizer, args["dataset"], args["num_samples"])

    dispatch = {
        "TreeFinder": lambda: run_tree_finder_mode(args, log, ds, model, tokenizer),
        "ContextCite": lambda: run_context_cite_mode(args, log, ds, model, tokenizer),
        "TracLLM": lambda: run_tracllm_mode(args, log, ds, model, tokenizer),
        "SelfCitation": lambda: run_self_citation_mode(args, log, ds, model, tokenizer),
        "necessity_sufficiency": lambda: run_necessity_sufficiency_mode(
            args, log, ds, model, tokenizer
        ),
        "topw_group_scores": lambda: run_topw_group_scores_mode(args, ds, log),
        "get_best_sentences": lambda: run_get_best_sentences_mode(args, log, ds),
        "llm_judge": lambda: run_llm_judge_mode(
            args, log, ds, model, tokenizer, top_x=args["top_x"]
        ),
    }

    if cmd.mode in dispatch:
        dispatch[cmd.mode]()
    elif cmd.mode == "all":
        for fn in [
            lambda: run_tree_finder_mode(args, log, ds, model, tokenizer),
            lambda: run_context_cite_mode(args, log, ds, model, tokenizer),
            lambda: run_tracllm_mode(args, log, ds, model, tokenizer),
            lambda: run_self_citation_mode(args, log, ds, model, tokenizer),
            lambda: run_necessity_sufficiency_mode(args, log, ds, model, tokenizer),
            lambda: run_aggregate_plots_mode(args, log),
            lambda: run_ranking_analysis_mode(args, log),
            lambda: run_topw_group_scores_mode(args, ds, log),
            lambda: plot_topw_group_scores(args, log),
            lambda: run_get_best_sentences_mode(args, log, ds),
            lambda: run_llm_judge_mode(
                args, log, ds, model, tokenizer, top_x=args["top_x"]
            ),
            lambda: plot_llm_judge_results(args, log),
        ]:
            fn()
        log.info("All modes completed!")

    log.info("Execution completed.")


if __name__ == "__main__":
    main()
