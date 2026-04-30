"""
Context Analysis System using Tree-based Rationale Extraction.

This module implements a hierarchical approach to identifying relevant context
chunks for question answering tasks, using significance scoring and tree-based
pruning to assign scores for each sentence.
"""

import logging
import heapq
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer
from .tree_utils import (
    AbstractFinder,
    AbstractFinderAPI,
    AbstractFinderTransformer,
    AbstractFinderVLLM,
)


class TreeFinder(AbstractFinder):
    """
    Hierarchical context finder that uses tree-based approach to identify
    relevant context chunks for question answering.

    This finder progressively narrows down context by merging sentences into
    chunks and evaluating their significance for answering the given question.
    """

    _supports_concurrent = False  # True for API backends (thread-safe, I/O-bound)

    def __init__(
        self,
        factor: int,
        topk: int,
        batch_size: int,
        expansion_budget: int = 500,
        logger: Optional[logging.Logger] = None,
        alpha: float = 0.5,
    ):
        """
        Initialize TreeFinder with configuration parameters.

        Args:
            factor (int): Initial chunking factor determining the number of chunks
            topk (int): Maximum number of chunks to retain
            batch_size (int): Batch size for model inference
            logger (logging.Logger): Logger instance for logging
            alpha (float, optional): Indicator for necessity vs sufficiency weighting,
                                     where 0.0 means only sufficiency, 1.0 means only necessity
                                     (useful when ablatting results). Defaults to 0.5.

        Raises:
            ValueError: If any of the parameters are out of valid ranges
        """

        if factor <= 0:
            raise ValueError("Factor must be positive")
        if topk <= 0:
            raise ValueError("Top-k must be positive")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        self.topk = topk
        self.factor = factor
        self.expansion_budget = expansion_budget
        self.batch_size = batch_size
        self.alpha = alpha
        self.logger = logger
        self.version = "v7"

        if self.logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.DEBUG)

        self.logger.info(
            f"Initialized TreeFinder with factor={factor}, "
            f"topk={topk}"
        )

    def _calculate_necessity_sufficiency(
        self,
        context_chunks: List[str],
        ablation_vector: torch.Tensor,
        inclusion_mapping: Dict[int, List[int]],
        question: str,
        precomputed_full_answer: Tuple[torch.Tensor, str, float],
        alpha: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate necessity and sufficiency scores for each context chunk.

        Args:
            context_chunks (List[str]): List of context chunk strings
            ablation_vector (torch.Tensor): Binary vector indicating which chunks to consider
            inclusion_mapping (Dict[int, List[int]]): Mapping from chunk indices to sentence indices
            question (str): The question to answer
            precomputed_full_answer (Tuple[torch.Tensor, str, float]): Precomputed answer data (tokens, answer, log_prob)
            log_prob_empty (float): Log probability of answer with empty context
            alpha (float): indicator for necessity vs sufficiency weighting,
                   where 0.0 means only sufficiency, 1.0 means only necessity
                   (useful when ablatting results)

        Returns:
            Tuple of (necessity_scores, sufficiency_scores) tensors

        Raises:
            ValueError: If inputs have incompatible dimensions
        """
        if len(context_chunks) != len(inclusion_mapping):
            raise ValueError(
                f"Context chunks ({len(context_chunks)}) and inclusion mapping "
                f"({len(inclusion_mapping)}) must have same length"
            )

        # Get indices of chunks that are included in ablation vector
        included_indices = [
            idx
            for idx in range(len(context_chunks))
            if ablation_vector[inclusion_mapping[idx][0]] == 1
        ]

        # Create identity matrix for included indices
        n_chunks = len(context_chunks)
        identity = torch.eye(n_chunks, dtype=torch.float32)

        solo_masks = identity[included_indices, :]

        return self._calculate_necessity_sufficiency_from_masks(
            solo_masks,
            question,
            context_chunks,
            precomputed_full_answer,
            alpha=alpha,
        )

    def _calculate_necessity_sufficiency_from_masks(
        self,
        solo_masks: torch.Tensor,
        question: str,
        context_chunks: List[str],
        precomputed_full_answer: Tuple[torch.Tensor, str, float],
        alpha: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate necessity and sufficiency scores using provided masks.

        Args:
            solo_masks (torch.Tensor): Binary masks for context chunks
            question (str): The question to answer
            context_chunks (List[str]): List of context chunk strings
            precomputed_full_answer (Tuple[torch.Tensor, str, float]): Precomputed answer data (tokens, answer, log_prob)
            log_prob_empty (float): Log probability of answer with empty context
            alpha (float, optional): indicator for necessity vs sufficiency weighting,
                                     where 0.0 means only sufficiency, 1.0 means only necessity
                                     (useful when ablatting results). Defaults to 0.25.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (necessity_scores, sufficiency_scores) tensors
        """

        log_probs_full = precomputed_full_answer[2]

        if alpha > 0.0:
            log_probs_removed = self.get_log_probs(
                1 - solo_masks,
                question,
                context_chunks,
                batch_size=self.batch_size,
                precomputed_answer=precomputed_full_answer,
            )[0]
            necessity_scores = log_probs_full + log_probs_removed

            if alpha == 1.0:
                return necessity_scores, torch.zeros_like(necessity_scores)
        else:
            necessity_scores = torch.zeros(solo_masks.shape[0], dtype=torch.float32)

        log_probs_solo = self.get_log_probs(
            solo_masks,
            question,
            context_chunks,
            batch_size=self.batch_size,
            precomputed_answer=precomputed_full_answer,
        )[0]

        sufficiency_scores = log_probs_full - log_probs_solo

        return necessity_scores, sufficiency_scores

    @staticmethod
    def _split_contiguous_indices(indices: List[int], branching_factor: int) -> List[List[int]]:
        """Split contiguous indices into up to branching_factor contiguous children."""
        if not indices:
            return []
        if branching_factor <= 1 or len(indices) <= 1:
            return [indices]

        step = max(1, (len(indices) + branching_factor - 1) // branching_factor)
        return [indices[i: i + step] for i in range(0, len(indices), step)]

    @staticmethod
    def _build_mask(num_sentences: int, selected_indices: List[int]) -> torch.Tensor:
        """Build a binary sentence mask for a subset of sentence indices."""
        mask = torch.zeros(num_sentences, dtype=torch.float32)
        if selected_indices:
            mask[selected_indices] = 1.0
        return mask

    def _compute_balanced_subset_scores(
        self,
        question: str,
        context: List[str],
        subset_indices_list: List[List[int]],
        precomputed_full_answer: Tuple[torch.Tensor, str, float],
    ) -> torch.Tensor:
        """
        Compute subset-level scores using the balanced necessity/sufficiency metric.

        score(C_i) = (necessity + sufficiency) / 2
               = (P(A|C) + P(A|C\\C_i) + P(A|C) - P(A|C_i)) / 2

        We use the model's configured probability scale (typically log-prob/logit),
        and keep the convention that lower values are better.
        """
        if not subset_indices_list:
            return torch.tensor([], dtype=torch.float32)

        num_sentences = len(context)
        solo_masks = torch.stack(
            [self._build_mask(num_sentences, idxs) for idxs in subset_indices_list],
            dim=0,
        )
        removed_masks = 1 - solo_masks

        full_score = precomputed_full_answer[2]
        removed_scores = self.get_log_probs(
            removed_masks,
            question,
            context,
            batch_size=self.batch_size,
            precomputed_answer=precomputed_full_answer,
        )[0]
        solo_scores = self.get_log_probs(
            solo_masks,
            question,
            context,
            batch_size=self.batch_size,
            precomputed_answer=precomputed_full_answer,
        )[0]

        return full_score + 0.5 * (removed_scores - solo_scores)

    def get_scores(self, question: str, context: List[str], precomputed_answer: Optional[Tuple[torch.Tensor, str, float]] = None) -> torch.Tensor:
        """
        Process single data instance end-to-end using A*-like best-first search.

        Args:
            question: The question to answer
            context: List of context sentences

        Returns:
            Tensor of significance scores for each sentence in context
        """

        if precomputed_answer is None:
            tokens, answer, initial_total_log_prob = self.generate_answer(
                context_sentences=context,
                question=question,
            )
        else:
            tokens, answer, initial_total_log_prob = precomputed_answer

        # fix for missing EOS
        tokens = torch.cat([tokens, torch.tensor([self.tokenizer.pad_token_id]).to(tokens.device)], dim=0)

        num_sentences = len(context)
        if num_sentences == 0:
            return torch.tensor([], dtype=torch.float32)

        precomputed_full_answer = (tokens, answer, initial_total_log_prob)

        # Lower is better for this score; initialize with +inf and fill leaves.
        sentence_scores = torch.full((num_sentences,), float("inf"), dtype=torch.float32)

        branching_factor = max(2, self.factor)
        retain_factor = max(1, min(self.topk, branching_factor))

        # Max-ablations is treated as a budget on expanded internal nodes.
        expansion_budget = self.expansion_budget // 2
        expanded_internal_nodes = 1

        # Min-heap ordered by A*-style normalized score density.
        frontier: List[Tuple[float, int, Tuple[int, ...], float]] = []
        push_counter = 0

        # Seed frontier with root children, so all candidates use the same metric.
        root_indices = list(range(num_sentences))
        root_children = self._split_contiguous_indices(root_indices, branching_factor)
        root_child_scores = self._compute_balanced_subset_scores(
            question,
            context,
            root_children,
            precomputed_full_answer,
        )

        for child_indices, child_score in zip(root_children, root_child_scores.tolist()):
            priority = child_score / max(1, len(child_indices))
            heapq.heappush(
                frontier,
                (priority, push_counter, tuple(child_indices), child_score),
            )
            push_counter += 1

            for idx in child_indices:
                sentence_scores[idx] = float(child_score)

        while frontier and expanded_internal_nodes < expansion_budget:
            _, _, node_indices_tuple, node_score = heapq.heappop(frontier)
            node_indices = list(node_indices_tuple)

            if len(node_indices) == 1:
                sentence_scores[node_indices[0]] = float(node_score)
                continue

            children = self._split_contiguous_indices(node_indices, branching_factor)
            if len(children) <= 1:
                # Degenerate split, score directly as leaf-equivalent.
                for idx in node_indices:
                    sentence_scores[idx] = float(node_score)
                continue

            child_scores = self._compute_balanced_subset_scores(
                question,
                context,
                children,
                precomputed_full_answer,
            )

            # Keep only the best children from this split, but selection is global via frontier.
            child_candidates = []
            for child_indices, child_score in zip(children, child_scores.tolist()):
                child_priority = child_score / max(1, len(child_indices))
                child_candidates.append((child_priority, child_indices, child_score))

                for idx in child_indices:
                    sentence_scores[idx] = float(child_score)

            child_candidates.sort(key=lambda x: x[0])
            for child_priority, child_indices, child_score in child_candidates[:retain_factor]:
                heapq.heappush(
                    frontier,
                    (child_priority, push_counter, tuple(child_indices), child_score),
                )
                push_counter += 1

            expanded_internal_nodes += len(children)

        print(f"Expanded {expanded_internal_nodes} internal nodes with branching factor {branching_factor} and retain factor {retain_factor}", flush=True)

        return sentence_scores, answer

    def get_necessity_sufficiency(
        self, question: str, context: List[str], indices: List[int] | List[List[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the necessity and sufficiency scores for a set of sentences.
        Args:
            question: The question to answer
            context: List of context sentences
            indices: List of sentence indices or list of list of sentence indices
        Returns:
            Tuple of (necessity_score, sufficiency_score)
        """

        tokens, answer, initial_total_log_prob = self.generate_answer(
            context_sentences=context,
            question=question,
        )

        # Construct the masks
        solo_masks = torch.zeros((len(indices), len(context)), dtype=torch.float32)
        for i, idx_list in enumerate(indices):
            solo_masks[i, idx_list] = 1.0

        # Calculate chunk significance
        return self._calculate_necessity_sufficiency_from_masks(
            solo_masks,
            question,
            context,
            precomputed_full_answer=(tokens, answer, initial_total_log_prob)
        )


class TreeFinderTransformer(TreeFinder, AbstractFinderTransformer):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        **kwargs,
    ):
        AbstractFinderTransformer.__init__(
            self, model_name, model_path, model=model, tokenizer=tokenizer
        )
        TreeFinder.__init__(self, **kwargs)


class TreeFinderAPI(TreeFinder, AbstractFinderAPI):
    def __init__(
        self, model_name: str, model_path: str, url: str, api_key: str, **kwargs
    ):
        AbstractFinderAPI.__init__(self, model_name, model_path, url, api_key)
        TreeFinder.__init__(self, **kwargs)


class TreeFinderVLLM(TreeFinder, AbstractFinderVLLM):
    def __init__(
        self,
        model_name: str,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        **kwargs,
    ):
        AbstractFinderVLLM.__init__(
            self, model_name, model_path, tensor_parallel_size, gpu_memory_utilization
        )
        TreeFinder.__init__(self, **kwargs)
