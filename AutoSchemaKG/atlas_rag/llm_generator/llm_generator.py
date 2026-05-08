

from typing import Optional, Union, Any

from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed, wait_exponential, wait_random, RetryCallState
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import traceback

from atlas_rag.llm_generator.prompt.rag_prompt import cot_system_instruction, cot_system_instruction_kg, cot_system_instruction_no_doc, prompt_template
from atlas_rag.llm_generator.prompt.lkg_prompt import ner_prompt, keyword_filtering_prompt, simple_ner_prompt
from atlas_rag.llm_generator.prompt.rag_prompt import filter_triple_messages
from atlas_rag.llm_generator.format.validate_json_output import *
from atlas_rag.llm_generator.format.validate_json_schema import filter_fact_json_schema, lkg_keyword_json_schema, ATLAS_SCHEMA
from atlas_rag.llm_generator.generation_config import GenerationConfig

from openai import OpenAI, AzureOpenAI, NOT_GIVEN, BadRequestError, RateLimitError
from transformers.pipelines import Pipeline
from transformers import AutoTokenizer

import json
import jsonschema
import time


def _count_tokens_api_safe(text: str) -> int:
    """
    Conservative token estimate for OpenAI-compatible APIs (often stricter than tiktoken for Qwen, etc.).
    """
    if not text:
        return 0
    try:
        import tiktoken

        n = len(tiktoken.get_encoding("cl100k_base").encode(text, disallowed_special=()))
    except Exception:
        n = max(1, len(text) // 3)
    return int(n * 1.18) + 16


def _truncate_string_to_token_budget_api_safe(s: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    lo, hi = 0, len(s)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        chunk = s[:mid]
        if _count_tokens_api_safe(chunk) <= max_tokens:
            best = chunk
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _truncate_context_to_fit_prompt(
    context: str,
    system_text: str,
    max_prompt_tokens: int,
    user_suffix: str,
) -> str:
    """
    Only shrink the context block; keep user_suffix (e.g. question + newlines) intact.
    """
    fixed = (
        _count_tokens_api_safe(system_text)
        + _count_tokens_api_safe(user_suffix)
        + 128
    )
    budget = max_prompt_tokens - fixed
    if budget <= 0:
        return ""
    if _count_tokens_api_safe(context) <= budget:
        return context
    lines = context.split("\n")
    lo, hi = 0, len(lines)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = "\n".join(lines[:mid])
        if _count_tokens_api_safe(cand) <= budget:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    if best:
        return best
    if lines:
        return _truncate_string_to_token_budget_api_safe(lines[0], budget)
    return ""


def serialize_openai_tool_call_message(message) -> dict:
    # Initialize the output dictionary
    serialized = {
        "role": message.role,
        "content": None if not message.content else message.content,
        "tool_calls": []
    }
    
    # Serialize each tool call
    for tool_call in message.tool_calls:
        serialized_tool_call = {
            "id": tool_call.id,
            "type": tool_call.type,
            "function": {
                "name": tool_call.function.name,
                "arguments": json.dumps(tool_call.function.arguments)
            }
        }
        serialized["tool_calls"].append(serialized_tool_call)
    
    return serialized

def print_retry(retry_state: RetryCallState):
    # Access the instance via retry_state.args[0] if method is bound
    instance = retry_state.args[0] if retry_state.args else None
    
    # Print the error that caused the retry
    exception = retry_state.outcome.exception()
    stack_trace = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    print(f"Error occurred: {type(exception).__name__}: {str(exception)}\nStack trace:\n{stack_trace}")

    if instance and hasattr(instance, 'retry_count'):
        instance.retry_count += 1
        print(f"Retrying {retry_state.fn.__name__}: attempt {retry_state.attempt_number}, total retries: {instance.retry_count}")
    else:
        print(f"Retrying {retry_state.fn.__name__}: attempt {retry_state.attempt_number}")

retry_decorator = retry(
    stop=(stop_after_delay(120) | stop_after_attempt(5)),  # Max 2 minutes or 5 attempts
    # wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(min=0, max=2),
    wait=wait_fixed(0),
    before_sleep=print_retry,
)

class LLMGenerator():
    """
    Unified LLM generator supporting multiple backends:
    
    Inference Types:
    - api: OpenAI-compatible API inference (OpenAI, Azure, Together AI, DeepInfra, vLLM server, etc.)
    - pipeline: HuggingFace Transformers pipeline
    - vllm: vLLM offline engine (local inference)
    - sglang: SGLang runtime (local inference)
    - tensorrt: TensorRT-LLM (local inference)
    
    Backend Types (for API inference):
    - openai: OpenAI API
    - azure: Azure OpenAI API
    - vllm: vLLM server (OpenAI-compatible)
    - sglang: SGLang server (OpenAI-compatible)
    - tensorrt: TensorRT-LLM server (OpenAI-compatible)
    - together: Together AI API
    - deepinfra: DeepInfra API
    - custom: Other OpenAI-compatible APIs
    """
    
    def __init__(self, client: Union[OpenAI, AzureOpenAI, Pipeline, Any], 
                 model_name: str, 
                 backend: str = 'openai', 
                 max_workers: int = 8,
                 default_config: Optional[GenerationConfig] = None):
        """
        Initialize LLM generator.
        
        Args:
            client: Client object (OpenAI, Pipeline, vLLM.LLM, SGLang runtime, TensorRT-LLM engine)
            model_name: Model identifier
            backend: Backend type ('openai', 'azure', 'vllm', 'sglang', 'tensorrt', 'together', 'deepinfra', 'custom')
            max_workers: Number of concurrent workers for API inference
            default_config: Default GenerationConfig for all generation calls (optional)
        """
        self.model_name = model_name
        self.client = client
        self.max_workers = max_workers
        self.backend = backend
        self.retry_count = 0
        
        # Set default generation config
        self.config = default_config if default_config is not None else GenerationConfig()
        
        # Determine inference type based on client
        if isinstance(client, (OpenAI, AzureOpenAI)):
            self.inference_type = "api"
            self.tokenizer = None
        elif isinstance(client, Pipeline):
            self.inference_type = "pipeline"
            self.tokenizer = client.tokenizer if hasattr(client, 'tokenizer') else None
        else:
            # Check for vLLM offline engine
            if hasattr(client, 'generate') and hasattr(client, 'get_tokenizer'):
                self.inference_type = "vllm"
                self.tokenizer = client.get_tokenizer()
            # Check for SGLang runtime
            elif hasattr(client, 'generate') and 'sglang' in str(type(client)).lower():
                self.inference_type = "sglang"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Check for TensorRT-LLM
            elif hasattr(client, 'generate') and 'tensorrt' in str(type(client)).lower():
                self.inference_type = "tensorrt"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                raise ValueError(
                    f"Unsupported client type: {type(client)}. "
                    "Please provide one of: OpenAI, AzureOpenAI, Pipeline, vLLM.LLM, SGLang runtime, or TensorRT-LLM engine."
                )
        
        print(f"Initialized LLMGenerator with inference_type='{self.inference_type}', backend='{self.backend}'")

    def _format_messages_for_local(self, messages):
        """Format messages for local inference engines that may need special handling."""
        if self.tokenizer is None:
            return messages
        
        # For local inference, apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            base_kw = dict(tokenize=False, add_generation_prompt=True)
            tmpl_kw = dict(self.config.chat_template_kwargs or {})
            try:
                if tmpl_kw:
                    return self.tokenizer.apply_chat_template(messages, **base_kw, **tmpl_kw)
                return self.tokenizer.apply_chat_template(messages, **base_kw)
            except TypeError:
                # Template or tokenizer does not accept kwargs (e.g. enable_thinking).
                try:
                    return self.tokenizer.apply_chat_template(messages, **base_kw)
                except Exception as e:
                    print(f"Warning: Failed to apply chat template: {e}")
                    return "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            except Exception as e:
                print(f"Warning: Failed to apply chat template: {e}")
                return "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return messages

    @retry_decorator
    def _api_inference(self, message, config: Optional[GenerationConfig] = None,
                           max_new_tokens=8192,
                           temperature = 0.7,
                           frequency_penalty = None,
                           repetition_penalty = None,
                           response_format = {"type": "text"},
                           return_text_only=True,
                           return_thinking=False,
                           reasoning_effort=None,
                           presence_penalty = None,
                           **kwargs):
        """
        API inference (OpenAI-compatible) with support for GenerationConfig or legacy parameters.
        Used for: OpenAI, Azure, vLLM server, SGLang server, TensorRT-LLM server, Together AI, etc.
        
        Args:
            message: The message(s) to send to the API
            config: GenerationConfig object (preferred). If provided, overrides other parameters.
            max_new_tokens: Legacy parameter (used if config is None)
            temperature: Legacy parameter (used if config is None)
            ... other legacy parameters ...
        """
        start_time = time.time()
        
        # Use GenerationConfig if provided, otherwise fall back to legacy parameters
        if config is not None:
            # Override config values with explicitly passed parameters
            if max_new_tokens != 8192:  # If not default, override
                config.max_tokens = max_new_tokens
            if temperature != 0.7:
                config.temperature = temperature
            if response_format is not None and response_format != {"type": "text"}:
                config.response_format = response_format
            if reasoning_effort is not None:
                config.reasoning_effort = reasoning_effort
            if return_thinking:
                if config.chat_template_kwargs is None:
                    config.chat_template_kwargs = {}
                config.chat_template_kwargs["enable_thinking"] = True
            
            # Convert config to OpenAI parameters based on backend
            api_params = config.to_openai_params(backend=self.backend)
            extra_body_params = config.to_extra_body(backend=self.backend)
        else:
            # Legacy parameter handling (mirror GenerationConfig.to_extra_body template fields)
            _et = False if reasoning_effort is None else True
            extra_body_params = {
                "chat_template_kwargs": {"enable_thinking": _et},
                "enable_thinking": _et,
            }
        
            # For pure OpenAI/Azure API, map repetition_penalty to frequency_penalty and presence_penalty
            if self.backend in ["openai", "azure"]:
                if repetition_penalty is not None:
                    # Split the effect between frequency and presence penalties
                    penalty_value = min(2.0, max(0.0, (repetition_penalty - 1.0) * 2.0))
                    if frequency_penalty is None:
                        frequency_penalty = penalty_value * 0.7  # 70% weight on frequency
                    if presence_penalty is None:
                        presence_penalty = penalty_value * 0.3   # 30% weight on presence
            else:
                # For vLLM, SGLang, TensorRT servers and other APIs, use repetition_penalty directly
                if repetition_penalty is not None:
                    extra_body_params["repetition_penalty"] = repetition_penalty
                    extra_body_params["early_stopping"] = config.early_stopping

            api_params = {
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "frequency_penalty": NOT_GIVEN if frequency_penalty is None else frequency_penalty,
                "presence_penalty": NOT_GIVEN if presence_penalty is None else presence_penalty,
                "response_format": response_format if response_format is not None else {"type": "text"},
                "timeout": 120,
                "reasoning_effort": NOT_GIVEN if reasoning_effort is None else reasoning_effort,
            }
        
        def _chat_completions_create(extra_body_arg):
            if extra_body_arg and extra_body_arg != NOT_GIVEN:
                return self.client.chat.completions.create(
                    model=self.model_name,
                    messages=message,
                    extra_body=extra_body_arg,
                    **api_params,
                )
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=message,
                **api_params,
            )

        try:
            response = _chat_completions_create(extra_body_params)
        except RateLimitError:
            # Never treat quota / 429 as "bad chat template kwargs" (message may still mention enable_thinking).
            raise
        except BadRequestError:
            # Only 400: some servers do not accept chat_template_kwargs / enable_thinking in extra_body.
            if not (
                isinstance(extra_body_params, dict)
                and (
                    extra_body_params.get("chat_template_kwargs")
                    or "enable_thinking" in extra_body_params
                )
            ):
                raise
            stripped = {
                k: v
                for k, v in extra_body_params.items()
                if k not in ("chat_template_kwargs", "enable_thinking")
            }
            print(
                "Warning: tokenizer template kwargs rejected by API (BadRequestError); "
                "retrying without chat_template_kwargs / enable_thinking."
            )
            response = _chat_completions_create(
                stripped if stripped else NOT_GIVEN
            )
        time_cost = time.time() - start_time
        content = response.choices[0].message.content
        if content is None and hasattr(response.choices[0].message, 'reasoning_content'):
            content = response.choices[0].message.reasoning_content
        validate_function = kwargs.get('validate_function', None)
        content = validate_function(content, **kwargs) if validate_function else content

        if '</think>' in content and not return_thinking:
            content = content.split('</think>')[-1].strip()
        else:
            if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content is not None and return_thinking:
                content = '<think>' + response.choices[0].message.reasoning_content + '</think>' + content
        

        if return_text_only:
            return content
        else:
            completion_usage_dict = response.usage.model_dump()
            completion_usage_dict['time'] = time_cost
            return content, completion_usage_dict

    @retry_decorator
    def _local_inference(self, message, config: Optional[GenerationConfig] = None,
                        max_new_tokens=8192,
                        temperature=0.7,
                        repetition_penalty=None,
                        return_text_only=True,
                        return_thinking=False,
                        **kwargs):
        """
        Local inference for vLLM offline, SGLang, or TensorRT-LLM.
        
        Args:
            message: The message(s) to send to the local engine
            config: GenerationConfig object (preferred)
            max_new_tokens: Legacy parameter
            temperature: Legacy parameter
            repetition_penalty: Legacy parameter
            return_text_only: Whether to return only text or include usage stats
            return_thinking: Whether to include thinking tokens
            **kwargs: Additional parameters
        """
        start_time = time.time()
        
        # Create config from legacy parameters if not provided
        if config is None:
            config = GenerationConfig(
                max_tokens=max_new_tokens,
                temperature=temperature,
                repetition_penalty=repetition_penalty
            )
        
        # Format messages for local inference
        if isinstance(message, list) and len(message) > 0 and isinstance(message[0], dict):
            # Chat format messages
            formatted_input = self._format_messages_for_local(message)
        else:
            formatted_input = message
        
        # Generate based on inference type
        if self.inference_type == "vllm":
            # vLLM offline engine
            from vllm import SamplingParams
            sampling_params = SamplingParams(**config.to_vllm_params())
            outputs = self.client.generate(formatted_input, sampling_params)
            content = outputs[0].outputs[0].text.strip()
            
        elif self.inference_type == "sglang":
            # SGLang runtime
            generation_params = config.to_sglang_params()
            outputs = self.client.generate(formatted_input, **generation_params)
            if isinstance(outputs, list):
                content = outputs[0].strip()
            else:
                content = outputs.strip()
                
        elif self.inference_type == "tensorrt":
            # TensorRT-LLM
            generation_params = config.to_tensorrt_params()
            outputs = self.client.generate(formatted_input, **generation_params)
            if isinstance(outputs, list):
                content = outputs[0].strip()
            else:
                content = outputs.strip()
        
        else:
            raise ValueError(f"Unsupported local inference type: {self.inference_type}")
        
        time_cost = time.time() - start_time
        
        # Apply validation if provided
        validate_function = kwargs.get('validate_function', None)
        if validate_function:
            content = validate_function(content, **kwargs)
        
        # Process thinking tags
        if '</think>' in content and not return_thinking:
            content = content.split('</think>')[-1].strip()
        
        if return_text_only:
            return content
        else:
            completion_usage_dict = {
                'completion_tokens': len(content.split()),
                'time': time_cost
            }
            return content, completion_usage_dict


    def generate_response(self, batch_messages, config: Optional[GenerationConfig] = None,
                             do_sample=None, max_new_tokens=None,
                             temperature=None, frequency_penalty=None, repetition_penalty=None,
                             response_format=None,
                             return_text_only=True, return_thinking=False, reasoning_effort=None, **kwargs):
        """
        Generate responses using self.config as default, with optional parameter overrides.
        Automatically routes to the appropriate inference method based on inference_type.
        
        Args:
            batch_messages: Single message (list of dict) or batch (list of list of dict)
            config: GenerationConfig object (overrides self.config completely if provided)
            do_sample: Override self.config.do_sample if provided
            max_new_tokens: Override self.config.max_tokens if provided
            temperature: Override self.config.temperature if provided
            frequency_penalty: Override self.config.frequency_penalty if provided
            repetition_penalty: Override self.config.repetition_penalty if provided
            response_format: Override self.config.response_format if provided
            reasoning_effort: Override self.config.reasoning_effort if provided
            return_text_only: Whether to return only text or include usage stats
            return_thinking: Whether to include thinking tokens
            **kwargs: Additional parameters
        """
        # Use provided config or create temporary config from self.config with overrides
        if config is None:
            # Create a copy of self.config
            from copy import deepcopy
            config = deepcopy(self.config)
            
            # Apply overrides if provided
            if max_new_tokens is not None:
                config.max_tokens = max_new_tokens
            if temperature is not None:
                config.temperature = temperature
                if temperature == 0.0:
                    config.do_sample = False
            if do_sample is not None:
                config.do_sample = do_sample
            if frequency_penalty is not None:
                config.frequency_penalty = frequency_penalty
            if repetition_penalty is not None:
                config.repetition_penalty = repetition_penalty
            if response_format is not None:
                config.response_format = response_format
            if reasoning_effort is not None:
                config.reasoning_effort = reasoning_effort
        
        # single = list of dict, batch = list of list of dict
        is_batch = isinstance(batch_messages[0], list)
        if not is_batch:
            batch_messages = [batch_messages]
        batch_size = len(batch_messages)
        results = [None] * len(batch_messages)
        to_process = list(range(len(batch_messages)))
        
        # Route to appropriate inference method based on inference_type
        if self.inference_type == "api":
            # OpenAI-compatible API inference (OpenAI, Azure, vLLM server, SGLang server, etc.)
            max_workers = self.max_workers
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                def process_message(i):
                    try:
                        return self._api_inference(
                            batch_messages[i], config=config,
                            return_text_only=return_text_only, 
                            return_thinking=return_thinking, 
                            **kwargs
                        )
                    except Exception as e:
                        print(f"API inference error: {e}")
                        return "[]"
                futures = [executor.submit(process_message, i) for i in to_process]
            for i, future in enumerate(futures):
                try:
                    results[i] = future.result()
                except Exception as e:
                    print(f"Future {i} failed: {str(e)}")
                    results[i] = '[]'

        elif self.inference_type in ["vllm", "sglang", "tensorrt"]:
            # Local inference engines (vLLM offline, SGLang, TensorRT-LLM)
            max_retries = kwargs.get('max_retries', 3)
            
            for i in to_process:
                retry_count = 0
                success = False
                
                while retry_count <= max_retries and not success:
                    try:
                        result = self._local_inference(
                            batch_messages[i],
                            config=config,
                            return_text_only=return_text_only,
                            return_thinking=return_thinking,
                            **kwargs
                        )
                        results[i] = result
                        success = True
                    except Exception as e:
                        retry_count += 1
                        if retry_count > max_retries:
                            print(f"Local inference failed for message {i} after {max_retries} retries: {e}")
                            results[i] = "[]" if return_text_only else ("[]", {'completion_tokens': 0, 'time': 0})
                        else:
                            print(f"Retry {retry_count}/{max_retries} for message {i}: {e}")

        elif self.inference_type == "pipeline":
            # HuggingFace Transformers pipeline
            max_retries = kwargs.get('max_retries', 3)
            start_time = time.time()
            
            # Convert config to HuggingFace parameters
            generation_kwargs = config.to_huggingface_params()
            
            # Initial processing of all messages
            responses = self.client(
                batch_messages,
                **generation_kwargs
            )
            time_cost = time.time() - start_time
            
            # Extract contents
            contents = [resp[0]['generated_text'].strip() for resp in responses]
            
            # Validate and collect failed indices
            validate_function = kwargs.get('validate_function', None)
            failed_indices = []
            for i, content in enumerate(contents):
                if validate_function:
                    try:
                        contents[i] = validate_function(content, **kwargs)
                    except Exception as e:
                        print(f"Validation failed for index {i}: {e}")
                        failed_indices.append(i)
            
            # Retry failed messages in batches
            for attempt in range(max_retries):
                if not failed_indices:
                    break
                print(f"Retry attempt {attempt + 1}/{max_retries} for {len(failed_indices)} failed messages")
                failed_messages = [batch_messages[i] for i in failed_indices]
                try:
                    retry_responses = self.client(
                        failed_messages,
                        **generation_kwargs
                    )
                    retry_contents = [resp[0]['generated_text'].strip() for resp in retry_responses]
                    
                    new_failed_indices = []
                    for j, i in enumerate(failed_indices):
                        try:
                            if validate_function:
                                retry_contents[j] = validate_function(retry_contents[j], **kwargs)
                            contents[i] = retry_contents[j]
                        except Exception as e:
                            print(f"Validation failed for index {i} on retry {attempt + 1}: {e}")
                            new_failed_indices.append(i)
                    failed_indices = new_failed_indices
                except Exception as e:
                    print(f"Batch retry {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        for i in failed_indices:
                            contents[i] = ""
            
            # Set remaining failed messages to ""
            for i in failed_indices:
                contents[i] = ""
            
            # Process thinking tags
            if not return_thinking:
                contents = [content.split('</think>')[-1].strip() if '</think>' in content else content for content in contents]
            
            if return_text_only:
                results = contents
            else:
                usage_dicts = [{
                    'completion_tokens': len(content.split()),
                    'time': time_cost / len(batch_messages)
                } for content in contents]
                results = list(zip(contents, usage_dicts))
        
        return results[0] if not is_batch else results

    # the function belows are all constructed for specific prompting and based on _api_inference and _local_inference

    def generate_cot(self, question, max_new_tokens=None):
        messages = [
            {"role": "system", "content": "".join(cot_system_instruction_no_doc)},
            {"role": "user", "content": question},
        ]
        return self.generate_response(messages, max_new_tokens=max_new_tokens)

    def generate_with_context(self, question, context, max_new_tokens=None, temperature=None):
        system_content = "".join(cot_system_instruction)
        ctx = context if isinstance(context, str) else "\n".join(str(x) for x in context)
        user_suffix = f"\n\n{question}\nThought:"
        max_in = getattr(self.config, "max_prompt_tokens", 30720)
        ctx = _truncate_context_to_fit_prompt(
            ctx,
            system_content,
            max_in,
            user_suffix,
        )
        print(f"{ctx}{user_suffix}")
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{ctx}{user_suffix}"},
        ]
        return self.generate_response(messages, max_new_tokens=max_new_tokens, temperature=temperature)

    def generate_with_context_one_shot(self, question, context, max_new_tokens=None, temperature=None):
        messages = deepcopy(prompt_template)
        messages.append(
            {"role": "user", "content": f"{context}\n\nQuestions:{question}\nThought:"},
        )
        return self.generate_response(messages, max_new_tokens=max_new_tokens, temperature=temperature)
    
    def generate_with_context_kg(self, question, context, max_new_tokens=None, temperature=None):
        system_content = "".join(cot_system_instruction_kg)
        ctx = context if isinstance(context, str) else "\n".join(str(x) for x in context)
        user_suffix = "\n\n" + question
        max_in = getattr(self.config, "max_prompt_tokens", 30720)
        ctx = _truncate_context_to_fit_prompt(
            ctx,
            system_content,
            max_in,
            user_suffix,
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{ctx}{user_suffix}"},
        ]
        return self.generate_response(messages, max_new_tokens=max_new_tokens, temperature=temperature)
        
    @retry_decorator
    def filter_triples_with_entity_event(self, question, triples, max_new_tokens=None, temperature=None):
        messages = deepcopy(filter_triple_messages)
        messages.append(
            {"role": "user", "content": f"""[ ## question ## ]]
        {question}

        [[ ## fact_before_filter ## ]]
        {triples}"""})
        try:
            validate_args = {
                "schema": filter_fact_json_schema,
                "fix_function": fix_filter_triplets,
            }
            response = self.generate_response(
                messages,
                max_new_tokens=max_new_tokens if max_new_tokens is not None else 4096,
                temperature=temperature if temperature is not None else 0.0,
                response_format={"type": "json_object"},
                validate_function=validate_output,
                **validate_args
            )
            return response
        except Exception as e:
            # If all retries fail, return the original triples
            return triples
        
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def large_kg_filter_keywords_with_entity(self, question, keywords, max_new_tokens=None, temperature=None):
        messages = deepcopy(keyword_filtering_prompt)
        
        messages.append({
            "role": "user",
            "content": f"""[[ ## question ## ]]
            {question}
            [[ ## keywords_before_filter ## ]]
            {keywords}"""
        })
        
        try:
            response = self.generate_response(
                messages, 
                response_format={"type": "json_object"}, 
                temperature=temperature if temperature is not None else 0.0,
                max_new_tokens=max_new_tokens if max_new_tokens is not None else 2048
            )
            
            # Validate and clean the response
            cleaned_data = validate_output(response, lkg_keyword_json_schema, fix_lkg_keywords)
            
            return cleaned_data['keywords']
        except Exception as e:
            return keywords
    
    def ner(self, text):
        messages = [
            {"role": "system", "content": "Please extract the entities from the following question and output them separated by comma, in the following format: entity1, entity2, ..."},
            {"role": "user", "content": f"Extract the named entities from: {text}"},
        ]
        return self.generate_response(messages)
    
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def large_kg_ner(self, text, simple_ner=False, max_new_tokens=None, temperature=None, frequency_penalty=None):
        if not simple_ner:
            messages = deepcopy(ner_prompt)
            messages.append(
                {
                    "role": "user", 
                    "content": f"[[ ## question ## ]]\n{text}" 
                }
            )
        else:
            messages = deepcopy(simple_ner_prompt)
            messages.append(
                {
                    "role": "user", 
                    "content": """
                    extracts named entities from given text.
                    Output them in Json format as follows:
                    {
                        "keywords": ["entity1", "entity2", ...]
                    }
                    Given text: 
                    """ + text
                }
            )
        validation_args = {
            "schema": lkg_keyword_json_schema,
            "fix_function": fix_lkg_keywords
        }
        # Generate raw response from LLM
        raw_response = self.generate_response(
            messages, 
            max_new_tokens=max_new_tokens if max_new_tokens is not None else 4096,
            temperature=temperature if temperature is not None else 0.7,
            frequency_penalty=frequency_penalty if frequency_penalty is not None else 1.1,
            response_format={"type": "json_object"},
            validate_function=validate_output,
            **validation_args
        )
        
        try:
            # Validate and clean the response
            cleaned_data = json_repair.loads(raw_response)
            return cleaned_data['keywords']
        
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            return []  # Fallback to empty list or raise custom exception
 
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def large_kg_tog_ner(self, text, max_new_tokens=None, temperature=None, frequency_penalty=None):
        messages = deepcopy(simple_ner_prompt)
        messages.append(
            {
                "role": "user", 
                "content": """
                extracts named entities from given text.
                Output them in Json format as follows:
                {
                    "keywords": ["entity1", "entity2", ...]
                }
                Given text: 
                """ + text
            }
        )
        # Generate raw response from LLM
        validation_args = {
            "schema": lkg_keyword_json_schema,
            "fix_function": fix_lkg_keywords
        }
        raw_response = self.generate_response(
            messages,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else 4096,
            temperature=temperature if temperature is not None else 0.7,
            frequency_penalty=frequency_penalty if frequency_penalty is not None else 1.1,
            response_format={"type": "json_object"},
            validate_function=validate_output,
            **validation_args
        )
        
        try:
            # Validate and clean the response
            cleaned_data = json_repair.loads(raw_response)
            return cleaned_data['keywords']
        
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            return []  # Fallback to empty list or raise custom exception

    def generate_with_react(self, question, context=None, max_new_tokens=None, search_history=None, logger=None):
        react_system_instruction = (
            'You are an advanced AI assistant that uses the ReAct framework to solve problems through iterative search. '
            'Follow these steps in your response:\n'
            '1. Thought: Think step by step and analyze if the current context is sufficient to answer the question. If not, review the current context and think critically about what can be searched to help answer the question.\n'
            '   - Break down the question into *1-hop* sub-questions if necessary (e.g., identify key entities like people or places before addressing specific events).\n'
            '   - Use the available context to make inferences about key entities and their relationships.\n'
            '   - If a previous search query (prefix with "Previous search attempt") was not useful, reflect on why and adjust your strategy—avoid repeating similar queries and consider searching for general information about key entities or related concepts.\n'
            '2. Action: Choose one of:\n'
            '   - Search for [Query]: If you need more information, specify a new query. The [Query] must differ from previous searches in wording and direction to explore new angles.\n'
            '   - No Action: If the current context is sufficient.\n'
            '3. Answer: Provide one of:\n'
            '   - A concise, definitive response as a noun phrase if you can answer.\n'
            '   - "Need more information" if you need to search.\n\n'
            'Format your response exactly as:\n'
            'Thought: [your reasoning]\n'
            'Action: [Search for [Query] or No Action]\n'
            'Answer: [concise noun phrase if you can answer, or "Need more information" if you need to search]\n\n'
        )
        
        # Build context with search history if available
        full_context = []
        if search_history:
            for i, (thought, action, observation) in enumerate(search_history):
                search_history_text = f"\nPrevious search attempt {i}:\n"
                search_history_text += f"{action}\n  Result: {observation}\n"
                full_context.append(search_history_text)
        if context:
            full_context_text = f"Current Retrieved Context:\n{context}\n"
            full_context.append(full_context_text)
        if logger:
            logger.info(f"Full context for ReAct generation: {full_context}")
        
        # Combine few-shot examples with system instruction and user query
        messages = [
            {"role": "system", "content": react_system_instruction},
            {"role": "user", "content": f"Search History:\n\n{''.join(full_context)}\n\nQuestion: {question}" 
            if full_context else f"Question: {question}"}
        ]
        if logger:
            logger.info(f"Messages for ReAct generation: {search_history}Question: {question}")
        return self.generate_response(messages, max_new_tokens=max_new_tokens)
  
    def triple_extraction(self, messages, result_schema, max_tokens=None, 
                     record=False, allow_empty=True, repetition_penalty=None):
        """
        Extract triples from messages with JSON validation using self.config as default.
        
        Args:
            messages: Single message (list of dict) or batch (list of list of dict)
            result_schema: JSON schema for validation
            max_tokens: Override self.config.max_tokens if provided
            repetition_penalty: Override self.config.repetition_penalty if provided
            record: Whether to return usage statistics
            allow_empty: Whether to allow empty results on validation failure
        """
        if isinstance(messages[0], dict):
            messages = [messages]
            
        validate_kwargs = {
            'schema': result_schema,
            'fix_function': fix_triple_extraction_response, # modify the fix according to provided schema
            'allow_empty': allow_empty
        }
        try:
            result = self.generate_response(
                messages, 
                max_new_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                # TODO: Update to support json_object for formatting with LLM
                # response_format={"type": "json_object"},
                validate_function=validate_output, 
                return_text_only=not record, 
                **validate_kwargs
            )
            return result
        except Exception as e:
            print(f"Triple extraction failed: {e}")
            # Return empty result if validation fails and allow_empty is True
            if allow_empty:
                if record:
                    return [], {'completion_tokens': 0, 'time': 0}
                else:
                    return []
            else:
                raise e