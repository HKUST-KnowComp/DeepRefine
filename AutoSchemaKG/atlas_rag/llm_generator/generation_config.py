from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import asyncio
from openai import NOT_GIVEN

@dataclass
class GenerationConfig:
    """
    Unified configuration for generation parameters across different backends.
    
    Supports: OpenAI, Azure OpenAI, vLLM, SGLang, TensorRT-LLM, HuggingFace Transformers,
    and third-party providers (Together AI, Anthropic, etc.)
    """
    
    # Core generation parameters (supported by all backends)
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    
    # Repetition control (backend-specific mapping)
    frequency_penalty: Optional[float] = None  # OpenAI, Azure, Together AI (-2.0 to 2.0)
    presence_penalty: Optional[float] = None   # OpenAI, Azure, Together AI (-2.0 to 2.0)
    repetition_penalty: Optional[float] = None # vLLM, SGLang, TensorRT-LLM, HF (typically 1.0-2.0)
    
    # Sampling parameters
    do_sample: bool = True
    seed: Optional[int] = None
    min_p: Optional[float] = None  # vLLM, SGLang (minimum probability threshold)
    
    # Stop sequences
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None  # vLLM, TensorRT-LLM
    include_stop_str_in_output: bool = False  # vLLM
    
    # Logprobs and token selection
    logprobs: Optional[int] = None  # OpenAI, vLLM (number of log probabilities to return)
    top_logprobs: Optional[int] = None  # OpenAI (0-20)
    echo: bool = False  # vLLM (echo back the prompt)
    
    # Length penalties
    length_penalty: Optional[float] = None  # HuggingFace, TensorRT-LLM (>1.0 encourages longer sequences)
    min_tokens: Optional[int] = None  # vLLM, SGLang
    
    # Beam search (HuggingFace, TensorRT-LLM)
    num_beams: int = 1
    early_stopping: bool = False
    
    # N-best generation
    n: int = 1  # OpenAI, vLLM (number of completions to generate)
    best_of: Optional[int] = None  # vLLM (generate best_of and return top n)
    
    # Advanced sampling (vLLM, SGLang)
    use_beam_search: bool = False
    ignore_eos: bool = False  # vLLM
    skip_special_tokens: bool = True  # vLLM, HF
    spaces_between_special_tokens: bool = True  # vLLM
    
    # Guided generation (vLLM, SGLang)
    guided_json: Optional[Union[str, Dict, Any]] = None  # vLLM, SGLang
    guided_regex: Optional[str] = None  # vLLM, SGLang
    guided_choice: Optional[List[str]] = None  # vLLM, SGLang
    guided_grammar: Optional[str] = None  # vLLM, SGLang
    guided_decoding_backend: Optional[str] = None  # vLLM: "outlines" or "lm-format-enforcer"
    guided_whitespace_pattern: Optional[str] = None  # vLLM
    
    # Truncation and padding (HuggingFace)
    truncation: bool = True
    padding: Union[bool, str] = False
    max_length: Optional[int] = None  # HuggingFace (different from max_tokens)
    
    # Response format (OpenAI, Together AI)
    response_format: Optional[Dict[str, str]] = None  # {"type": "json_object"} or {"type": "text"}
    
    # Model-specific (OpenAI o1/o3 models)
    reasoning_effort: Optional[str] = None  # "low", "medium", "high"
    
    # Prompt caching (Anthropic, some APIs)
    prompt_lookup_num_tokens: Optional[int] = None  # vLLM
    
    # Logit bias (OpenAI, Together AI)
    logit_bias: Optional[Dict[int, float]] = None
    
    # User identifier (OpenAI, for abuse monitoring)
    user: Optional[str] = None
    
    # API / chat: estimated max input tokens for the full prompt (system + user).
    # Stay below provider context (e.g. 32k) to absorb chat-template drift vs tiktoken. Raise for long-context models.
    max_prompt_tokens: int = 30000
    
    # Timeout
    timeout: float = 120.0
    
    # Stream response
    stream: bool = False
    
    # Kwargs forwarded to the inference server's tokenizer.apply_chat_template(...)
    # (OpenAI-compatible: extra_body["chat_template_kwargs"], e.g. vLLM / Qwen).
    # Not generation/sampling parameters.
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    
    # Tools and function calling (OpenAI, Together AI)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict]] = None
    
    def to_openai_params(self, backend: str = "openai") -> Dict[str, Any]:
        """
        Convert to OpenAI API parameters.
        Also works for Azure OpenAI and OpenAI-compatible APIs.
        """
        params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        # Handle repetition_penalty mapping for non-OpenAI backends
        if backend == "openai" and self.repetition_penalty is not None and self.frequency_penalty is None:
            penalty_value = min(2.0, max(0.0, (self.repetition_penalty - 1.0) * 2.0))
            params["frequency_penalty"] = penalty_value * 0.7
            params["presence_penalty"] = penalty_value * 0.3
        else:
            if self.frequency_penalty is not None:
                params["frequency_penalty"] = self.frequency_penalty
            if self.presence_penalty is not None:
                params["presence_penalty"] = self.presence_penalty
        
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.stop is not None:
            params["stop"] = self.stop
        if self.seed is not None:
            params["seed"] = self.seed
        if self.logprobs is not None:
            params["logprobs"] = self.logprobs
        if self.top_logprobs is not None:
            params["top_logprobs"] = self.top_logprobs
        if self.n > 1:
            params["n"] = self.n
        if self.response_format is not None:
            params["response_format"] = self.response_format
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort
        if self.logit_bias is not None:
            params["logit_bias"] = self.logit_bias
        if self.user is not None:
            params["user"] = self.user
        if self.tools is not None:
            params["tools"] = self.tools
        if self.tool_choice is not None:
            params["tool_choice"] = self.tool_choice
        if self.stream:
            params["stream"] = self.stream
        
        return params
    
    def to_vllm_params(self) -> Dict[str, Any]:
        """
        Convert to vLLM SamplingParams.
        Supports both vLLM server (OpenAI-compatible) and offline LLM.
        """
        params = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        if self.n > 1:
            params["n"] = self.n
        if self.best_of is not None:
            params["best_of"] = self.best_of
        if self.repetition_penalty is not None:
            params["repetition_penalty"] = self.repetition_penalty
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.top_k is not None and self.top_k > 0:
            params["top_k"] = self.top_k
        if self.min_p is not None:
            params["min_p"] = self.min_p
        if self.min_tokens is not None:
            params["min_tokens"] = self.min_tokens
        if self.use_beam_search:
            params["use_beam_search"] = self.use_beam_search
        if self.length_penalty is not None:
            params["length_penalty"] = self.length_penalty
        if self.stop is not None:
            params["stop"] = self.stop if isinstance(self.stop, list) else [self.stop]
        if self.stop_token_ids is not None:
            params["stop_token_ids"] = self.stop_token_ids
        if self.ignore_eos:
            params["ignore_eos"] = self.ignore_eos
        if not self.skip_special_tokens:
            params["skip_special_tokens"] = self.skip_special_tokens
        if not self.spaces_between_special_tokens:
            params["spaces_between_special_tokens"] = self.spaces_between_special_tokens
        if self.logprobs is not None:
            params["logprobs"] = self.logprobs
        if self.prompt_lookup_num_tokens is not None:
            params["prompt_lookup_num_tokens"] = self.prompt_lookup_num_tokens
        if self.seed is not None:
            params["seed"] = self.seed
        if self.include_stop_str_in_output:
            params["include_stop_str_in_output"] = self.include_stop_str_in_output
        
        # Guided generation
        if self.guided_json is not None:
            params["guided_json"] = self.guided_json
        if self.guided_regex is not None:
            params["guided_regex"] = self.guided_regex
        if self.guided_choice is not None:
            params["guided_choice"] = self.guided_choice
        if self.guided_grammar is not None:
            params["guided_grammar"] = self.guided_grammar
        if self.guided_decoding_backend is not None:
            params["guided_decoding_backend"] = self.guided_decoding_backend
        if self.guided_whitespace_pattern is not None:
            params["guided_whitespace_pattern"] = self.guided_whitespace_pattern
        
        return params
    
    def to_sglang_params(self) -> Dict[str, Any]:
        """
        Convert to SGLang parameters.
        SGLang supports similar parameters to vLLM.
        """
        params = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            params["repetition_penalty"] = self.repetition_penalty
        if self.frequency_penalty is not None:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty is not None:
            params["presence_penalty"] = self.presence_penalty
        if self.stop is not None:
            params["stop"] = self.stop if isinstance(self.stop, list) else [self.stop]
        if self.seed is not None:
            params["seed"] = self.seed
        if not self.skip_special_tokens:
            params["skip_special_tokens"] = self.skip_special_tokens
        if self.ignore_eos:
            params["ignore_eos"] = self.ignore_eos
        
        # Guided generation (SGLang supports similar to vLLM)
        if self.guided_json is not None:
            params["guided_json"] = self.guided_json
        if self.guided_regex is not None:
            params["guided_regex"] = self.guided_regex
        if self.guided_choice is not None:
            params["guided_choice"] = self.guided_choice
        
        return params
    
    def to_tensorrt_params(self) -> Dict[str, Any]:
        """
        Convert to TensorRT-LLM parameters.
        """
        params = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            params["repetition_penalty"] = self.repetition_penalty
        if self.length_penalty is not None:
            params["length_penalty"] = self.length_penalty
        if self.num_beams > 1:
            params["num_beams"] = self.num_beams
        if self.stop is not None:
            params["end_strings"] = self.stop if isinstance(self.stop, list) else [self.stop]
        if self.stop_token_ids is not None:
            params["stop_words_list"] = self.stop_token_ids
        if self.seed is not None:
            params["random_seed"] = self.seed
        if self.early_stopping:
            params["early_stopping"] = self.early_stopping
        
        return params
    
    def to_huggingface_params(self) -> Dict[str, Any]:
        """
        Convert to HuggingFace Transformers generation parameters.
        For use with pipeline or model.generate().
        """
        params = {
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample if self.temperature > 0 else False,
        }
        
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.repetition_penalty is not None:
            params["repetition_penalty"] = self.repetition_penalty
        if self.length_penalty is not None:
            params["length_penalty"] = self.length_penalty
        if self.num_beams > 1:
            params["num_beams"] = self.num_beams
        if self.early_stopping:
            params["early_stopping"] = self.early_stopping
        if self.seed is not None:
            params["seed"] = self.seed
        if self.stop is not None:
            # HuggingFace uses eos_token_id or stopping_criteria
            params["stop_strings"] = self.stop if isinstance(self.stop, list) else [self.stop]
        if self.n > 1:
            params["num_return_sequences"] = self.n
        if self.max_length is not None:
            params["max_length"] = self.max_length
        if not self.skip_special_tokens:
            params["skip_special_tokens"] = self.skip_special_tokens
        
        # Pipeline-specific
        params["return_full_text"] = False
        params["truncation"] = self.truncation
        params["padding"] = self.padding
        
        return params
    
    def to_together_params(self) -> Dict[str, Any]:
        """
        Convert to Together AI parameters.
        Together AI uses OpenAI-compatible API but supports additional parameters.
        """
        params = self.to_openai_params(backend="together")
        
        # Together AI specific parameters
        if self.repetition_penalty is not None:
            params["repetition_penalty"] = self.repetition_penalty
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.min_p is not None:
            params["min_p"] = self.min_p
        
        return params
    
    def to_extra_body(self, backend: str) -> Dict[str, Any]:
        """
        Get extra_body parameters for OpenAI-compatible APIs.
        Used for passing backend-specific parameters through OpenAI client.
        """
        extra_body = {}
        
        if backend == "vllm":
            if self.repetition_penalty is not None:
                extra_body["repetition_penalty"] = self.repetition_penalty
            if self.min_p is not None:
                extra_body["min_p"] = self.min_p
            if self.guided_json is not None:
                extra_body["guided_json"] = self.guided_json
            if self.guided_regex is not None:
                extra_body["guided_regex"] = self.guided_regex
            if self.guided_choice is not None:
                extra_body["guided_choice"] = self.guided_choice
        
        if self.chat_template_kwargs is not None:
            # Shallow copy so callers mutating the request path cannot corrupt the config dict.
            extra_body["chat_template_kwargs"] = dict(self.chat_template_kwargs)
            # Some gateways report errors as `parameter.enable_thinking` and expect a top-level
            # field (in addition to chat_template_kwargs for the tokenizer).
            if "enable_thinking" in self.chat_template_kwargs:
                extra_body["enable_thinking"] = self.chat_template_kwargs["enable_thinking"]
        
        return extra_body if extra_body else NOT_GIVEN
