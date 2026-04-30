from typing import Dict, List, Optional, Tuple
import torch

from abc import ABC, abstractmethod

try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
except ImportError:
    print("vLLM library not found. Please install vLLM to use AbstractFinderVLLM.")


from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


DEFAULT_GENERATE_KWARGS_VLLM = {
    "max_new_tokens": 128,
    "do_sample": False,
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,  # this is for vLLM
    "repetition_penalty": 1.0,
}
DEFAULT_GENERATE_KWARGS = {
    "max_new_tokens": 128,
    "do_sample": False,
    "repetition_penalty": 1.0,
    "temperature": None,
    "top_p": None,
    "top_k": None,
}
DEFAULT_PROMPT_TEMPLATE = "###Instruction:\n Respond to the question based on the context provided. The answer must be concise and accurate.\n###Context:\n{context}\n###Question:\n{query}"


class AbstractFinder(ABC):
    def __init__(self, model_name: str, model_path: str = ""):
        self.model_name = model_name
        self.model_path = model_path
        self.prompt_format = DEFAULT_PROMPT_TEMPLATE

        # Will be set by child classes
        self.tokenizer = None
        self.device = None
        self.gen_kwargs = None

        self.num_calls = 0

    def get_and_reset_num_calls(self) -> int:
        """Get the number of calls made to the model and reset the counter."""
        num_calls = self.num_calls
        self.num_calls = 0
        return num_calls

    def generate_answer(
        self, context_sentences: List[str], question: str
    ) -> Tuple[torch.Tensor, str, float]:
        prompt = self.prompt_format.format(
            context=" ".join(context_sentences), query=question
        )
        return self.generate_answer_with_prompt(prompt)

    @abstractmethod
    def generate_answer_with_prompt(
        self, prompt: str
    ) -> Tuple[torch.Tensor, str, float]:
        """Generate answer with prompt and return tokens, text, and log probability."""
        pass

    @abstractmethod
    def compute_answer_probability(
        self,
        batch_prompts: List[List[Dict[str, str]]],
        answer: str,
        tokens: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute probability of answer given batch of prompts."""
        pass

    def get_log_probs(
        self,
        masks: torch.Tensor,
        question: str,
        context_sentences: List[str],
        batch_size: int,
        precomputed_answer: Optional[Tuple[torch.Tensor, str, float]] = None,
    ):
        """Get a list of log probabilities for mask of the context sentences."""

        logit_probs = torch.zeros(masks.shape[0], dtype=torch.float32)

        if precomputed_answer:
            tokens, answer, initial_total_log_prob = precomputed_answer
        else:
            tokens, answer, initial_total_log_prob = self.generate_answer(
                context_sentences, question
            )

        for i in range(0, masks.shape[0], batch_size):
            # Calculate actual batch size to avoid index out of bounds
            actual_batch_size = min(batch_size, masks.shape[0] - i)

            batch_prompts = [
                [
                    {
                        "role": "user",
                        "content": self.prompt_format.format(
                            context=" ".join(
                                context_sentences[j]
                                for j in range(len(context_sentences))
                                if masks[i + k][j]
                            ),
                            query=question,
                        ),
                    }
                ]
                for k in range(actual_batch_size)
            ]

            logits = self.compute_answer_probability(
                batch_prompts, answer, tokens, actual_batch_size
            )

            logit_probs[i : i + actual_batch_size] = logits.cpu()

        return logit_probs, answer, len(tokens), initial_total_log_prob

    @staticmethod
    def compute_log_probs(logits, labels):
        """Compute log-probabilities of the target tokens."""
        batch_size, seq_length, vocab_size = logits.shape
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Gather the log-prob of the correct tokens
        target_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return target_log_probs

    @staticmethod
    def compute_logit_probs(logits, labels):
        """Compute logit - logsumexp(all logits) for target tokens."""
        batch_size, seq_length, vocab_size = logits.shape
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return target_log_probs  # These are actually log-probs, not "logit probs"

    @abstractmethod
    def get_scores(
        self, question: str, context: List[str], precomputed_answer: Optional[Tuple]
    ) -> torch.Tensor:
        """Process a single data instance end-to-end."""
        pass


class AbstractFinderTransformer(AbstractFinder):
    """Implementation using local transformer models."""

    def __init__(
        self,
        model_name: str,
        model_path: str = "",
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        super().__init__(model_name, model_path)

        if model is not None and tokenizer is not None:
            # Initialize from provided model and tokenizer
            self.model = model
            self.tokenizer = tokenizer
            self.device = model.device
            self.gen_kwargs = DEFAULT_GENERATE_KWARGS.copy()

            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            self.gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            self.gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
            return

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path + model_name)
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path + model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )
        self.model.eval()

        self.device = self.model.device
        self.gen_kwargs = DEFAULT_GENERATE_KWARGS.copy()
        self.gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        self.gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

    def generate_answer_with_prompt(
        self, prompt: str
    ) -> Tuple[torch.Tensor, str, float]:
        message = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        tokenized_chat = self.tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=False,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=tokenized_chat.input_ids.to(self.device),
                return_dict_in_generate=True,
                output_scores=True,
                attention_mask=tokenized_chat.attention_mask.to(self.device),
                **self.gen_kwargs,
            )

        self.num_calls += 1

        # Get only the generated tokens (after the prompt)
        generated_token_ids = outputs.sequences[0, tokenized_chat.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(
            generated_token_ids, skip_special_tokens=True
        )

        # Calculate log probabilities using the defined function compute_logit_probs
        log_probs = self.compute_log_probs(
            torch.stack(outputs.scores, dim=1), generated_token_ids.unsqueeze(0)
        )

        return generated_token_ids, generated_text, log_probs.sum().item()

    def compute_answer_probability(
        self,
        batch_prompts: List[List[Dict[str, str]]],
        answer: str,
        tokens: torch.Tensor,
        batch_size: int,
    ):
        self.tokenizer.padding_side = "left"
        batch_prompt_inputs = self.tokenizer.apply_chat_template(
            batch_prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            truncation=False,
            enable_thinking=False,
        )

        expanded_answer = tokens.expand(batch_size, -1)
        input_ids_batch = torch.cat(
            [batch_prompt_inputs.input_ids.to(self.device), expanded_answer], dim=1
        )
        attention_mask_batch = torch.cat(
            [batch_prompt_inputs.attention_mask.to(self.device), torch.ones_like(expanded_answer, dtype=torch.long)], dim=1
        )

        with torch.no_grad():
            batch_outputs = self.model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        self.num_calls += batch_size

        answer_logits = batch_outputs.logits[:, batch_prompt_inputs.input_ids.size(1) - 1: -1, :]

        log_probs = self.compute_log_probs(
            answer_logits, input_ids_batch[:, batch_prompt_inputs.input_ids.size(1):]
        )

        return log_probs.sum(dim=1)


class AbstractFinderAPI(AbstractFinder):
    """Implementation using OpenAI-compatible API endpoints."""

    def __init__(
        self,
        model_name: str,
        model_path: str = "",
        url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
    ):
        super().__init__(model_name, model_path)

        print(f"Connecting to OpenAI API at {url} with API key {api_key}")
        self.client = OpenAI(base_url=url, api_key=api_key, timeout=None)

        # Use tokenizer for encoding/decoding but not for inference
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path + model_name if model_path else model_name
        )
        self.tokenizer.padding_side = "left"

        self.device = "cpu"  # API doesn't use local device
        self.gen_kwargs = DEFAULT_GENERATE_KWARGS_VLLM.copy()

    def generate_answer_with_prompt(
        self, prompt: str
    ) -> Tuple[torch.Tensor, str, float]:
        message = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        tokenized_chat = self.tokenizer.apply_chat_template(
            message,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
            enable_thinking=False,
        )

        response = self.client.completions.create(
            model=self.model_name,
            prompt=tokenized_chat,
            max_tokens=self.gen_kwargs["max_new_tokens"],
            top_p=self.gen_kwargs["top_p"],
            temperature=self.gen_kwargs["temperature"],
            logprobs=1,
            extra_body={
                "top_k": self.gen_kwargs["top_k"],
                "repetition_penalty": self.gen_kwargs["repetition_penalty"],
            },
        )

        # Get guided response for consistent log probabilities
        response = self.client.completions.create(
            model=self.model_name,
            prompt=tokenized_chat,
            top_p=self.gen_kwargs["top_p"],
            temperature=self.gen_kwargs["temperature"],
            logprobs=1,
            max_tokens=self.gen_kwargs["max_new_tokens"],
            extra_body={
                "guided_choice": [response.choices[0].text],
                "guided_decoding_backend": "outlines",
                "top_k": self.gen_kwargs["top_k"],
                "repetition_penalty": self.gen_kwargs["repetition_penalty"],
            },
        )

        log_prob = sum(t for t in response.choices[0].logprobs.token_logprobs)
        response_text = response.choices[0].text

        tokens = self.tokenizer.encode(
            response_text, add_special_tokens=False, return_tensors="pt"
        ).squeeze(0)

        return tokens, response_text, log_prob

    def compute_answer_probability(
        self,
        batch_prompts: List[List[Dict[str, str]]],
        answer: str,
        tokens: torch.Tensor,
        batch_size: int,
    ):
        self.tokenizer.padding_side = "left"
        batch_prompt_inputs = self.tokenizer.apply_chat_template(
            batch_prompts,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors=None,
            padding=True,
            truncation=False,
            enable_thinking=False,
        )

        batch_responses = self.client.completions.create(
            model=self.model_name,
            prompt=batch_prompt_inputs,
            top_p=self.gen_kwargs["top_p"],
            temperature=self.gen_kwargs["temperature"],
            logprobs=1,
            max_tokens=self.gen_kwargs["max_new_tokens"],
            extra_body={
                "guided_choice": [answer],
                "guided_decoding_backend": "outlines",
                "top_k": self.gen_kwargs["top_k"],
                "repetition_penalty": self.gen_kwargs["repetition_penalty"],
            },
        )

        return torch.tensor(
            [
                sum(t for t in batch_responses.choices[k].logprobs.token_logprobs)
                for k in range(batch_size)
            ]
        )


class AbstractFinderVLLM(AbstractFinder):
    """Implementation using vLLM engine directly."""

    def __init__(
        self,
        model_name: str,
        model_path: str = "",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        super().__init__(model_name, model_path)

        print("Using vLLM engine for inference")

        # Initialize vLLM engine
        self.llm = LLM(
            model=model_path + model_name if model_path else model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="float16",
        )

        # Use the tokenizer from vLLM
        self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer.padding_side = "left"

        self.device = "cuda"  # vLLM uses GPU
        self.gen_kwargs = DEFAULT_GENERATE_KWARGS_VLLM.copy()

        # Create sampling params
        self.sampling_params = SamplingParams(
            max_tokens=self.gen_kwargs["max_new_tokens"],
            temperature=self.gen_kwargs["temperature"],
            top_p=self.gen_kwargs["top_p"],
            top_k=self.gen_kwargs["top_k"],
            repetition_penalty=self.gen_kwargs["repetition_penalty"],
            logprobs=1,
        )

    def generate_answer_with_prompt(
        self, prompt: str
    ) -> Tuple[torch.Tensor, str, float]:
        message = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        tokenized_chat = self.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        outputs = self.llm.generate(
            prompts=[tokenized_chat], sampling_params=self.sampling_params
        )
        output = outputs[0]

        generated_text = output.outputs[0].text

        # Calculate log probability from logprobs
        log_prob = sum(
            list(token.values())[0].logprob for token in output.outputs[0].logprobs
        )

        # Tokenize the generated text
        tokens = self.tokenizer.encode(
            generated_text, add_special_tokens=False, return_tensors="pt"
        ).squeeze(0)

        return tokens.cpu(), generated_text, log_prob

    def compute_answer_probability(
        self,
        batch_prompts: List[List[Dict[str, str]]],
        answer: str,
        tokens: torch.Tensor,
        batch_size: int,
    ):
        # Convert batch prompts to strings
        batch_prompt_strings = []
        for prompt_messages in batch_prompts:
            prompt_string = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            batch_prompt_strings.append(prompt_string)

        guided_decoding_params = GuidedDecodingParams(
            choice=[answer],
            backend="outlines",
        )

        # Create sampling params for guided generation
        guided_sampling_params = SamplingParams(
            max_tokens=len(tokens),
            temperature=0.0,  # Deterministic for probability calculation
            logprobs=1,
            prompt_logprobs=1,
            guided_decoding=guided_decoding_params,
        )

        # Process in batches to limit memory usage
        all_log_probs = []
        for i in range(0, len(batch_prompt_strings), batch_size):
            # Get batch slice
            batch_slice = batch_prompt_strings[i : i + batch_size]

            # Generate with the specific answer as constraint
            outputs = self.llm.generate(batch_slice, guided_sampling_params)

            batch_log_probs = []
            for output in outputs:
                # Calculate log probability from logprobs
                log_prob = sum(
                    list(token.values())[0].logprob
                    for token in output.outputs[0].logprobs
                )
                batch_log_probs.append(log_prob)

            all_log_probs.extend(batch_log_probs)

        return torch.tensor(all_log_probs)
