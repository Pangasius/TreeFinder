"""
Context Analysis System using Tree-based Rationale Extraction.

This module implements a hierarchical approach to identifying relevant context
chunks for question answering tasks, using significance scoring and tree-based
pruning to assign scores for each sentence.
"""

import logging
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer
from tree_utils import (
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

    def __init__(
        self,
        factor: int,
        threshold: float,
        cum_prob: float,
        topk: int,
        batch_size: int,
        logger: logging.Logger,
        alpha: float = 0.25,
    ):
        """
        Initialize TreeFinder with configuration parameters.

        Args:
            factor (int): Initial chunking factor determining the number of chunks
            threshold (float): Significance threshold for filtering chunks
            cum_prob (float): Cumulative probability threshold for top-p filtering
            topk (int): Maximum number of chunks to retain
            batch_size (int): Batch size for model inference
            logger (logging.Logger): Logger instance for logging
            alpha (float, optional): Indicator for necessity vs sufficiency weighting,
                                     where 0.0 means only sufficiency, 1.0 means only necessity
                                     (useful when ablatting results). Defaults to 0.25.

        Raises:
            ValueError: If any of the parameters are out of valid ranges
        """

        if factor <= 0:
            raise ValueError("Factor must be positive")
        if not 0 <= cum_prob <= 1:
            raise ValueError("Cumulative probability must be between 0 and 1")
        if topk <= 0:
            raise ValueError("Top-k must be positive")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")

        self.threshold = threshold
        self.cum_prob = cum_prob
        self.topk = topk
        self.factor = factor
        self.batch_size = batch_size
        self.alpha = alpha
        self.logger = logger
        self.version = "v2"

        self.logger.info(
            f"Initialized TreeFinder with factor={factor}, "
            f"threshold={threshold}, cum_prob={cum_prob}, topk={topk}"
        )

    def _calculate_necessity_sufficiency(
        self,
        context_chunks: List[str],
        ablation_vector: torch.Tensor,
        inclusion_mapping: Dict[int, List[int]],
        question: str,
        precomputed_full_answer: Tuple[torch.Tensor, str, float],
        log_prob_empty: float,
        alpha: float = 0.25,
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
            log_prob_empty,
            alpha=alpha,
        )

    def _calculate_necessity_sufficiency_from_masks(
        self,
        solo_masks: torch.Tensor,
        question: str,
        context_chunks: List[str],
        precomputed_full_answer: Tuple[torch.Tensor, str, float],
        log_prob_empty: float,
        alpha: float = 0.25,
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
        if alpha > 0.0:
            log_probs_removed = self.get_log_probs(
                1 - solo_masks,
                question,
                context_chunks,
                batch_size=self.batch_size,
                precomputed_answer=precomputed_full_answer,
            )[0]
            necessity_scores = log_probs_removed - log_prob_empty

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

        log_pro_full = precomputed_full_answer[2]
        sufficiency_scores = log_pro_full - log_probs_solo

        return necessity_scores, sufficiency_scores

    def _calculate_chunk_significance(
        self,
        context_chunks: List[str],
        ablation_vector: torch.Tensor,
        inclusion_mapping: Dict[int, List[int]],
        question: str,
        precomputed_full_answer: Tuple[torch.Tensor, str, float],
        log_prob_empty: float,
    ) -> torch.Tensor:
        """
        Calculate significance of each context chunk using batch processing.

        Args:
            context_chunks: List of context chunk strings
            ablation_vector: Binary vector indicating which chunks to consider
            inclusion_mapping: Mapping from chunk indices to sentence indices
            question: The question to answer
            precomputed_answer: Precomputed answer data (tokens, answer, log_prob)
            log_prob_empty: Log probability of answer with empty context

        Returns:
            Tensor of significance scores for each chunk
        """

        necessity_scores, sufficiency_scores = self._calculate_necessity_sufficiency(
            context_chunks,
            ablation_vector,
            inclusion_mapping,
            question,
            precomputed_full_answer,
            log_prob_empty,
            self.alpha,
        )

        significance_scores_expanded = torch.full(
            (len(context_chunks),),
            float("-inf"),
            dtype=torch.float32,
        )

        combined_score = (
            self.alpha * necessity_scores + (1 - self.alpha) * sufficiency_scores
        )

        # Map scores back to expanded tensor (exactly like original)
        idx_og = 0
        for idx in range(len(context_chunks)):
            if ablation_vector[inclusion_mapping[idx][0]] == 0:
                continue
            significance_scores_expanded[idx] = combined_score[idx_og]
            idx_og += 1

        return significance_scores_expanded

    def _filter_children(self, significance_scores: torch.Tensor) -> torch.Tensor:
        """
        Extract rationale chunks based on significance scores and thresholds.

        Applies three filtering criteria:
        1. Threshold-based filtering
        2. Top-p (cumulative probability) filtering
        3. Top-k filtering

        Args:
            significance_scores: Tensor of significance scores

        Returns:
            Binary mask indicating selected rationale chunks
        """
        if len(significance_scores) == 0:
            return torch.tensor([], dtype=torch.int32)

        mask = torch.ones_like(significance_scores, dtype=torch.int32)
        mask = mask & (significance_scores != float("-inf"))

        if (mask & (significance_scores < self.threshold)).sum() > 0:
            mask = mask & (significance_scores < self.threshold)

        # Apply softmax for probability-based filtering
        softmax_scores = torch.nn.functional.softmax(significance_scores, dim=0)

        # Apply top-p filtering
        sorted_indices = torch.argsort(softmax_scores, descending=False)

        # Apply top-k filtering
        if len(sorted_indices) > self.topk + (len(mask) - mask.sum()):
            mask[sorted_indices[self.topk + (len(mask) - mask.sum()) :]] = 0

        cumulative_sum = torch.cumsum(softmax_scores[sorted_indices], dim=0)
        cutoff_indices = torch.where(cumulative_sum > self.cum_prob)[0]

        if len(cutoff_indices) > 0:
            cutoff_idx = cutoff_indices[0].item() + 1
            mask[sorted_indices[cutoff_idx:]] = 0

        return mask.cpu()

    def _merge_context(
        self, sentences: List[str], factor: int
    ) -> Tuple[List[str], Dict[int, List[int]]]:
        """
        Merge context sentences into chunks based on configured factor.

        Args:
            sentences: List of sentence strings
            factor: Chunking factor determining the number of chunks

        Returns:
            Tuple of (chunk_list, inclusion_mapping)
            - chunk_list: List of merged chunks
            - inclusion_mapping: Dict mapping chunk index to sentence indices

        Raises:
            ValueError: If factor is non-positive
        """
        if factor <= 0:
            raise ValueError("Factor must be positive")
        if not sentences:
            return [], {}

        chunks = []
        current_chunk = ""
        inclusion = {}
        total_length = sum(len(sentence) for sentence in sentences)

        # Handle case where factor is too large
        if total_length // factor == 0:
            self.logger.debug("Factor too large, treating each sentence as chunk")
            for i, sentence in enumerate(sentences):
                chunks.append(sentence.strip())
                inclusion[i] = [i]
            return chunks, inclusion

        inclusion_counter = 0
        chunk_size_threshold = total_length // factor

        for sentence_idx, sentence in enumerate(sentences):
            if inclusion_counter not in inclusion:
                inclusion[inclusion_counter] = []

            inclusion[inclusion_counter].append(sentence_idx)
            current_chunk += sentence

            if len(current_chunk) > chunk_size_threshold:
                chunks.append(current_chunk.strip())
                current_chunk = ""
                inclusion_counter += 1

        # Add remaining content as final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks, inclusion

    def _update_ablation_vector(
        self,
        ablation_vector: torch.Tensor,
        rationale_mask: torch.Tensor,
        inclusion_mapping: Dict[int, List[int]],
    ) -> None:
        """
        Update ablation vector based on rationale mask.

        Args:
            ablation_vector: Current ablation vector
            rationale_mask: Mask indicating selected rationale chunks
            inclusion_mapping: Mapping from chunk to sentence indices
        """

        for idx, mask_value in enumerate(rationale_mask):
            if mask_value == 1:
                continue

            for included_chunk in inclusion_mapping[idx]:
                ablation_vector[included_chunk] &= mask_value

    def _update_chunk_significance(
        self,
        significance_scores: torch.Tensor,
        original_scores: torch.Tensor,
        inclusion_mapping: Dict[int, List[int]],
    ) -> None:
        """
        Update significance scores based on rationale mask.

        Args:
            significance_scores: Current significance scores
            original_scores: Original significance scores to update
            inclusion_mapping: Mapping from chunk to sentence indices
        """

        for idx in range(len(significance_scores)):
            if significance_scores[idx].item() == float("-inf"):
                continue
            if len(original_scores) != len(significance_scores):
                for included_chunk in inclusion_mapping[idx]:
                    original_scores[included_chunk] = significance_scores[idx].item()
            else:
                original_scores[idx] = significance_scores[idx].item()

    def get_scores(self, question: str, context: List[str]) -> torch.Tensor:
        """
        Process single data instance end-to-end.

        Args:
            question: The question to answer
            context: List of context sentences

        Returns:
            Tensor of significance scores for each sentence in context
        """
        tokens, answer, initial_total_log_prob = self.generate_answer(
            context_sentences=context,
            question=question,
        )

        empty_prompt = [
            [
                {
                    "role": "user",
                    "content": self.prompt_format.format(
                        context="",
                        query=question,
                    ),
                }
            ]
        ]

        log_prob_empty = self.compute_answer_probability(
            batch_prompts=empty_prompt, answer=answer, tokens=tokens, batch_size=1
        ).item()

        # Initialize tree search
        ablation_vector = torch.ones(len(context), dtype=torch.int32)
        significance_scores = torch.ones(len(context), dtype=torch.float32) * float(
            "inf"
        )
        previous_ablation_vector = ablation_vector.clone()
        factor = self.factor
        done = False

        _significance_cache = {}

        # Tree-based rationale extraction
        while not done:
            total_length = sum(len(sentence) for sentence in context)

            # Split context into chunks
            rationale_chunks, inclusion_mapping = self._merge_context(context, factor)

            # Check termination conditions
            if total_length // factor == 0 or len(rationale_chunks) == len(context):
                self.logger.debug("Reached sentence-level granularity")
                done = True

            # locally disable some computation using the cache
            masked_ablation_vector = ablation_vector.clone()
            for idx, chunk_indices in inclusion_mapping.items():
                cache_idx = (idx, tuple(chunk_indices))
                if cache_idx in _significance_cache:
                    for chunk_idx in chunk_indices:
                        if chunk_idx < len(masked_ablation_vector):
                            masked_ablation_vector[chunk_idx] = 0

            # Verify if any chunks have valid sentences
            valid_chunks = any(
                any(
                    masked_ablation_vector[sentence_idx] == 1
                    for sentence_idx in chunk_indices
                )
                for chunk_indices in inclusion_mapping.values()
            )

            if masked_ablation_vector.sum() == 0 or not valid_chunks:
                self.logger.debug("No chunks left to process, stopping early")
                break

            # Calculate chunk significance
            significance_scores_new = self._calculate_chunk_significance(
                rationale_chunks,
                masked_ablation_vector,
                inclusion_mapping,
                question,
                log_prob_empty=log_prob_empty,
                precomputed_full_answer=(tokens, answer, initial_total_log_prob),
            )

            # Create a cache for inclusion mapping elements to significance scores
            for idx, chunk_indices in inclusion_mapping.items():
                cache_idx = (idx, tuple(chunk_indices))
                if significance_scores_new[idx] == float("-inf"):
                    continue
                if cache_idx not in _significance_cache:
                    if idx < len(significance_scores_new):
                        _significance_cache[cache_idx] = significance_scores_new[
                            idx
                        ].clone()
                else:
                    if idx < len(significance_scores_new):
                        significance_scores_new[idx] = _significance_cache[cache_idx]

            # Extract rationale mask
            rationale_mask = self._filter_children(significance_scores_new)

            # Update significance scores
            self._update_chunk_significance(
                significance_scores_new, significance_scores, inclusion_mapping
            )

            # Update ablation vector
            self._update_ablation_vector(
                ablation_vector, rationale_mask, inclusion_mapping
            )

            # Early stopping if too few chunks remain
            if sum(ablation_vector) < 2:
                if sum(ablation_vector) == 0:
                    ablation_vector = previous_ablation_vector
                break

            # Prepare for next iteration
            factor *= self.factor / self.topk
            previous_ablation_vector = ablation_vector.clone()

        return significance_scores

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

        empty_prompt = [
            [
                {
                    "role": "user",
                    "content": self.prompt_format.format(
                        context="",
                        query=question,
                    ),
                }
            ]
        ]

        log_prob_empty = self.compute_answer_probability(
            batch_prompts=empty_prompt, answer=answer, tokens=tokens, batch_size=1
        ).item()

        # Construct the masks
        solo_masks = torch.zeros((len(indices), len(context)), dtype=torch.float32)
        for i, idx_list in enumerate(indices):
            solo_masks[i, idx_list] = 1.0

        # Calculate chunk significance
        return self._calculate_necessity_sufficiency_from_masks(
            solo_masks,
            question,
            context,
            precomputed_full_answer=(tokens, answer, initial_total_log_prob),
            log_prob_empty=log_prob_empty,
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
