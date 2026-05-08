# Utils
# Define the LLMGenerator class for generating responses from the LLM
# 

import json
from typing import List, Union
from openai import OpenAI, AzureOpenAI, NOT_GIVEN, AsyncOpenAI
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed, wait_exponential, wait_random
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import asyncio
from transformers.pipelines import Pipeline
import jsonschema
from transformers import AutoTokenizer
from sglang.srt.sampling.sampling_params import SamplingParams
import torch
import traceback

import time
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

class LLMGenerator():
    def __init__(self, client, model_name, backend = 'verl'):
        self.model_name = model_name
        self.client : OpenAI  = client
        if isinstance(client, (OpenAI,AzureOpenAI,AsyncOpenAI)):
            self.inference_type = "openai"
        if backend == 'verl':
            self.inference_type = "verl"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    async def _api_inference(self, message, max_new_tokens=8192,
                             temperature=0.7,
                             frequency_penalty=None,
                             response_format=None,
                             return_text_only=True,
                             return_thinking=False,
                             reasoning_effort=None,
                             **kwargs):
        start_time = time.time()
        serialize_message = deepcopy(message)
        for i, msg in enumerate(serialize_message):
            if isinstance(msg, dict):
                if 'tool_calls' in msg and msg['tool_calls'] is not None:
                    serialize_message[i] = serialize_openai_tool_call_message(msg)
            elif hasattr(msg, 'tool_calls') and msg.tool_calls is not None:
                serialize_message[i] = serialize_openai_tool_call_message(msg)
        prompt_logprobs = kwargs.get('prompt_logprobs', False)
        if prompt_logprobs:
            max_new_tokens = 0
            response = await self.client.completions.create(
                model=self.model_name,
                prompt=self.tokenizer.apply_chat_template(serialize_message),
                echo=True,
                logprobs=1,
                max_tokens=max_new_tokens,
            )
            return response
        logprobs=kwargs.get('logprobs', False)
        top_logprobs=NOT_GIVEN if not logprobs else 0
        # Do not send response_format to avoid triggering structured outputs in vLLM
        response = await self.client.chat.completions.create( 
            model=self.model_name,
            messages=serialize_message,
            max_tokens=max_new_tokens,
            temperature=temperature,
            frequency_penalty=NOT_GIVEN if frequency_penalty is None else frequency_penalty,
            response_format=NOT_GIVEN,
            timeout=120,
            reasoning_effort=NOT_GIVEN if reasoning_effort is None else reasoning_effort,      
            logprobs=logprobs,
            top_logprobs=top_logprobs,     
        )
        time_cost = time.time() - start_time
        content = response.choices[0].message.content
        if content is None and hasattr(response.choices[0].message, 'reasoning_content'):
            content = response.choices[0].message.reasoning_content
        validate_function = kwargs.get('validate_function', None)
        content = validate_function(content, **kwargs) if validate_function else content

        if '</think>' in content and not return_thinking:   # remove the thinking content
            content = content.split('</think>')[-1].strip()
        else:  # maintain the thinking content
            if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content is not None and return_thinking:
                content = '<think>' + response.choices[0].message.reasoning_content + '</think>' + content

        if return_text_only:
            return content
        else:
            return response
    @torch.no_grad()
    async def _verl_inference(self, message, max_new_tokens=8192,
                              temperature=0.0,
                              frequency_penalty=None,
                              return_text_only=True,
                              **kwargs):
        serialize_message = deepcopy(message)
        for i, msg in enumerate(serialize_message):
            if isinstance(msg, dict):
                if 'tool_calls' in msg and msg['tool_calls'] is not None:
                    serialize_message[i] = serialize_openai_tool_call_message(msg)
            elif hasattr(msg, 'tool_calls') and msg.tool_calls is not None:
                serialize_message[i] = serialize_openai_tool_call_message(msg)
        prompt_ids = self.tokenizer.apply_chat_template(serialize_message, tokenize=True, add_generation_prompt=True)
        sampling_params = {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "frequency_penalty": frequency_penalty if frequency_penalty is not None else 0.0,
        }
        return_log_probs = kwargs.get("return_log_prob", False)
        request_id = str(time.time())
        results = []

        result = await self.client.async_generate(input_ids=prompt_ids, sampling_params=sampling_params, return_logprob=False)
        results.append(result)
        
        # Handle different response structures from verl engine
        if results:
            last_result = results[-1]
            if 'text' in last_result:
                output_text = last_result['text']
            elif 'content' in last_result:
                output_text = last_result['content']
            elif 'output_text' in last_result:
                output_text = last_result['output_text']
            else:
                # Try to decode from output_ids if available
                if 'output_ids' in last_result:
                    output_ids = last_result['output_ids']
                    output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                else:
                    print(f"Warning: Unexpected result structure from verl engine: {last_result.keys()}")
                    output_text = ""
        else:
            output_text = ""
        content = output_text
        if return_text_only:
            return content
        else:
            return results[-1] if results else None        

    async def generate_response(self, batch_messages, do_sample=True, max_new_tokens=8192,
                                temperature=0.7, frequency_penalty=None, response_format={"type": "text"},
                                return_text_only=True, return_thinking=False, reasoning_effort=None, **kwargs):
        if temperature == 0.0:
            do_sample = False
        is_batch = isinstance(batch_messages[0], list)
        if not is_batch:
            batch_messages = [batch_messages]
        results = [None] * len(batch_messages)
        to_process = list(range(len(batch_messages)))

        if self.inference_type == "openai":
            async def process_message(i):
                try:
                    return await self._api_inference(
                        batch_messages[i], max_new_tokens, temperature,
                        frequency_penalty, response_format, return_text_only, return_thinking, reasoning_effort, **kwargs
                    )
                except Exception as e:
                    print(f"Error processing message {i}: {e} batch_messages: {batch_messages[i]}")
                    return ""
            tasks = [process_message(i) for i in to_process]
            results = await asyncio.gather(*tasks)
        elif self.inference_type == "verl":
            async def process_message(i):
                try:
                    return await self._verl_inference(
                        batch_messages[i], max_new_tokens, temperature,
                        frequency_penalty, return_text_only, **kwargs
                    )
                except Exception as e:
                    print(f"Error processing message {i}: {e}, batch_messages: {batch_messages[i]}")
                    traceback.print_exc()
                    return ""
            tasks = [process_message(i) for i in to_process]
            results = await asyncio.gather(*tasks)
        else:
            raise ValueError("Unsupported inference type")

        return results[0] if not is_batch else results
