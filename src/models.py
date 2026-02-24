"""
LLM Model Factory: Unified interface for commercial and open-source models.

Provides a single call_model() function that handles OpenAI (GPT), Anthropic (Claude), and local HuggingFace models (via Unsloth).
Standardizes responses and tracks costs.
"""

import os

# Disable HuggingFace symlink warning on Windows
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import time
from pathlib import Path
from typing import Any
from dataclasses import dataclass

from src.utils import calculate_cost

# Load environment variables for API keys
from dotenv import load_dotenv
load_dotenv()


# Model cache: stores loaded models to avoid reloading on every call
_MODEL_CACHE: dict[str, Any] = {}  # Key: model_name, Value: (model, tokenizer) tuple

# Helper function to enable/disable offline mode
def _set_offline_mode(enabled: bool) -> None:
    """
    Control HuggingFace offline mode.

    Args:
        enabled: If True, force offline mode. If False, allow online access.
    """
    if enabled:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'

        # Disable network requests at huggingface_hub level
        try:
            import huggingface_hub
            huggingface_hub.constants.HF_HUB_OFFLINE = True
        except (ImportError, AttributeError):
            pass

    else:
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
        os.environ.pop('HF_DATASETS_OFFLINE', None)

        try:
            import huggingface_hub
            huggingface_hub.constants.HF_HUB_OFFLINE = False
        except (ImportError, AttributeError):
            pass


@dataclass
class ModelResponse:
    """
    Standardized response structure from any LLM.

    Attributes:
        text: The generated text response
        input_tokens: Number of tokens in the prompt
        output_tokens: Number of tokens in the response
        model_name: Identifier of the model used
        tool_calls: List of tool calls if model used tools, empty list otherwise
        response_time: Time taken for the API call in seconds
        cost: Cost of the API call in USD (0.0 for Open-Source models)
    """
    text: str
    input_tokens: int
    output_tokens: int
    model_name: str
    tool_calls: list[dict[str, Any]]
    response_time: float
    cost: float


# Open-source models available through Unsloth on HuggingFace
UNSLOTH_MODELS: list[str] = [

    # Small models (3B)
    # "unsloth/Llama-3.2-3B-Instruct",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    # "unsloth/Qwen2.5-3B-Instruct",
    "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",

    # Medium models (7-9B)
    # "unsloth/Qwen2.5-7B-Instruct",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    # "unsloth/gemma-2-9b-it",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    # "unsloth/Meta-Llama-3.1-8B-Instruct",
    # "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", # Too large for 4GB GPU - causes paging errors
    # "unsloth/DeepSeek-R1-Distill-Qwen-7B",
    # "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"

    # Large models (>12B)
    # "unsloth/Qwen2.5-14B-Instruct",
    "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
    # "unsloth/Mistral-Nemo-Instruct-2407", (12B)
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    # "unsloth/Mistral-Small-Instruct-2409", (22B)
    # "unsloth/Mistral-Small-Instruct-2409-bnb-4bit",
]

# Commercial API providers
PAID_MODELS: list[str] = [
    "gpt-4o-mini",
    "claude-3-haiku-20240307"
]


def call_model(model_name: str, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> ModelResponse:
    """
    Call any LLM (OpenAI, Anthropic, or Unsloth) with unified interface.

    Args:
        model_name: Model identifier from UNSLOTH_MODELS or PAID_MODELS
        prompt: User message/prompt to send to the model
        max_tokens: Maximum tokens to generate in response
        temperature: Sampling temperature (0.0 to 1.0)

    Returns:
        ModelResponse with text, token counts, and tool calls

    Raises:
        ValueError: If model_name is not recognized
    """
    # Determine which provider handles this model
    if model_name in PAID_MODELS:
        if model_name.startswith("gpt-"):
            return _call_openai(model_name, prompt, max_tokens, temperature)
        elif model_name.startswith("claude-"):
            return _call_anthropic(model_name, prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported API-based model prefix: {model_name}")
    elif model_name in UNSLOTH_MODELS:
        return _call_unsloth(model_name, prompt, max_tokens, temperature)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def _call_openai(model_name: str, prompt: str, max_tokens: int, temperature: float) -> ModelResponse:
    """
    Call OpenAI API.

    Args:
        model_name: OpenAI model identifier (e.g., 'gpt-4o-mini')
        prompt: User message to send
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        ModelResponse with OpenAI's actual response data

    Raises:
        ImportError: If openai package not installed
        Exception: If API key missing or API call fails
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("OpenAI package not installed. Run: pip install openai")

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OPENAI_API_KEY not found in environment variables")

    # Initialize client
    client = OpenAI(api_key=api_key)

    # Measure response time
    start_time = time.time()

    # Make API call
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature
    )

    response_time = time.time() - start_time

    # Extract response data
    message = response.choices[0].message
    text = message.content or ""
    tool_calls = []

    # Check if model used tools
    if message.tool_calls:
        tool_calls = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                }
            }
            for tc in message.tool_calls
        ]

    # Calculate cost
    cost = calculate_cost(model_name=model_name, input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

    return ModelResponse(
        text=text,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        model_name=model_name,
        tool_calls=tool_calls,
        response_time=response_time,
        cost=cost
    )


def _call_anthropic(model_name: str, prompt: str, max_tokens: int, temperature: float) -> ModelResponse:
    """
    Call Anthropic API.

    Args:
        model_name: Anthropic model identifier (e.g., 'claude-3-5-haiku')
        prompt: User message to send
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        ModelResponse with Anthropic's actual response data

    Raises:
        ImportError: If anthropic package not installed
        Exception: If API key missing or API call fails
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        raise ImportError("Anthropic package not installed. Run: pip install anthropic")

    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise Exception("ANTHROPIC_API_KEY not found in environment variables")

    # Initialize client
    client = Anthropic(api_key=api_key)

    # Measure response time
    start_time = time.time()

    # Make API call
    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}]
    )

    response_time = time.time() - start_time

    # Extract response data
    text = ""
    tool_calls = []

    # Claude returns content as a LIST of blocks (text, tool_use, etc.)
    # We loop to handle multiple blocks and filter by type
    for block in response.content:
        if block.type == "text":
            text += block.text
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "type": "tool_use",
                "function": {
                    "name": block.name,
                    "arguments": block.input
                }
            })

    # Calculate cost
    cost = calculate_cost(model_name=model_name, input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)

    return ModelResponse(
        text=text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model_name=model_name,
        tool_calls=tool_calls,
        response_time=response_time,
        cost=cost
    )


def _call_unsloth(model_name: str, prompt: str, max_tokens: int, temperature: float) -> ModelResponse:
    """
    Call Unsloth/HuggingFace model locally. Tries to load from cache first. If not found, downloads automatically.
    Loads model on first call, reuses from cache on subsequent calls.

    Args:
        model_name: HuggingFace model identifier (e.g., 'unsloth/Llama-3.2-3B-Instruct-bnb-4bit')
        prompt: User message to send
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature

    Returns:
        ModelResponse with model's response data

    Note:
        First call downloads the model (~2-8GB), subsequent calls use cached version.
        Requires GPU for reasonable speed (CPU inference is very slow).
    """
    # Get cache directory
    cache_dir = os.path.join('hf_cache', 'hub')

    # Check if model exists in cache
    model_cache_path = Path(cache_dir) / f"models--{model_name.replace('/', '--')}"
    use_local_only = model_cache_path.exists()

    # Set offline mode BEFORE importing transformers
    _set_offline_mode(use_local_only)

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError:
        raise ImportError("Transformers/torch not installed. Run: pip install transformers torch")

    # Set the device (GPU if available, else CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")
    # if device == "cuda":
    #     print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model only if not already in cache
    if model_name not in _MODEL_CACHE:
        # Verify offline mode is set (Only print when actually loading)
        if use_local_only:
            print(f"Loading from cache: {model_cache_path}")
        else:
            print(f"Model not in cache, downloading from HuggingFace...")


        # Load model, tokenizer using the OLD SCHOOL way (reliable but too slow)
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=use_local_only
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,  # Automatically use GPU if available
            dtype=torch.float16,  # Use half precision for efficiency
            cache_dir=cache_dir,  # Use local hf_cache
            local_files_only=use_local_only  # local_files_only=use_local_only
        )

        # # For proper 4-bit loading and 2-3x faster inference, switch to:
        # from unsloth import FastLanguageModel
        # model, tokenizer = FastLanguageModel.from_pretrained(
        #     model_name=model_name,
        #     max_seq_length=2048,
        #     load_in_4bit=True,  # ← properly loads 4-bit weights
        #     local_files_only=use_local_only,
        #     cache_dir=cache_dir,
        # )
        # FastLanguageModel.for_inference(model)
        # Requires: GPU with 5GB+ VRAM and unsloth installed.
        # Current machine (GTX 1050 Ti, 4GB) cannot run 7B+ models locally.
        # Run on Google Colab or EuropeanHPC instead.

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Store in cache
        _MODEL_CACHE[model_name] = (model, tokenizer)

    # Retrieve from cache
    model, tokenizer = _MODEL_CACHE[model_name]

    # Measure response time
    start_time = time.time()

    # Tokenize input with attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, return_attention_mask=True).to(device)
    input_token_count = inputs.input_ids.shape[1]

    # Generate response
    with torch.no_grad():  # Disable gradient calculation for inference

        # do_sample controls randomness:
        # - do_sample=True + temperature > 0: varied responses ("Hello!", "Hi!", "Hey!")
        # - do_sample=False (temperature=0): same response always ("Hello!", "Hello!", "Hello!")

        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id
        )

    response_time = time.time() - start_time

    # Decode response (contains: prompt + generated_text)
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the output to get only the response
    response_text = full_output[len(prompt):].strip()

    # Calculate output tokens
    output_token_count = outputs.shape[1] - input_token_count

    # Cost is always 0.0 for local models
    return ModelResponse(
        text=response_text,
        input_tokens=input_token_count,
        output_tokens=output_token_count,
        model_name=model_name,
        tool_calls=[],  # Tool calling not implemented for local models yet
        response_time = response_time,
        cost=0.0
    )


def clear_model_cache() -> None:
    """
    Clear all loaded models from cache and free GPU memory.
    Call between models in experiments to avoid memory overflow.
    """
    global _MODEL_CACHE

    try:
        import torch
        for model_name, (model, _) in _MODEL_CACHE.items():
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    _MODEL_CACHE = {}
    # print("Model cache cleared")

