"""Model Exploration Script - Inspect raw API responses before ModelResponse wrapping."""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
from anthropic import Anthropic
from src.utils import print_separator

from dotenv import load_dotenv
load_dotenv()


def explore_openai_raw() -> None:
    """Explore raw OpenAI API response structure."""
    print_separator("RAW OPENAI RESPONSE STRUCTURE")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    prompt = "Say love in 5 different languages."
    print(f"\nPrompt: {prompt}\n")

    print("Calling OpenAI API...")
    raw_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=20,
        temperature=0.7
    )

    print("\n" + "=" * 60)
    print("\nRAW RESPONSE OBJECT\n")
    print(f"Type: {type(raw_response)}")
    print(f"\nFull object response:\n{raw_response}\n")
    print("=" * 60)

    print("\nUNPACKING STRUCTURE\n")
    # print(dir(raw_response))  # Print all attribute names

    # Step 1: Get all top-level fields
    print(f"Top-level fields: {list(raw_response.model_dump().keys())}")

    # Step 2: Convert to dict and inspect each field's type
    response_dict = raw_response.model_dump()
    print("\nField Types")
    for key, value in response_dict.items():
        print(f"  {key}: {type(value)}")

    print("\n" + "=" * 60)

    # Step 3: Dive into content
    print(f"\nDive into 'choices':\n{raw_response.choices}\n")
    print(f"choices: {type(raw_response.choices)} with {len(raw_response.choices)} item(s)")
    if raw_response.choices:
        first_choice = raw_response.choices[0]
        print(f"  choices[0]: {type(first_choice)}")
        print(f"  choices[0] fields: {list(first_choice.model_dump().keys())}")

        # Dive into message
        print(f"\nDive into 'message':\n{first_choice.message}\n")
        print(f"  choices[0].message: {type(first_choice.message)}")
        print(f"  choices[0].message fields: {list(first_choice.message.model_dump().keys())}")

        # Message content
        print(f"\n  Dive into 'content':")
        print(f"    message.content type: {type(first_choice.message.content)} = {first_choice.message.content}")

        print(f"\n  Other 'message' fields:")
        print(f"    message.role type: {type(first_choice.message.role)} = {first_choice.message.role}")
        print(f"    message.tool_calls: {type(first_choice.message.tool_calls)}")

    print("\n" + "=" * 60)

    # Step 4: Usage inspection
    print(f"\nDive into 'usage':\n{raw_response.usage}\n")
    print(f"usage: {type(raw_response.usage)}")
    print(f"usage fields: {list(raw_response.usage.model_dump().keys())}")
    for field, value in raw_response.usage.model_dump().items():
        print(f"  {field}: {type(value)}")

    print("\n" + "=" * 60)

    # Step 5: Simple fields
    print("\nSimple Fields")
    print(f"id: {type(raw_response.id)} = {raw_response.id}")
    print(f"model: {type(raw_response.model)} = {raw_response.model}")
    print(f"created: {type(raw_response.created)} = {raw_response.created}")
    print(f"object: {type(raw_response.object)} = {raw_response.object}")

    print("\n" + "=" * 60)
    print("\nOpenAI raw exploration complete!")


def explore_anthropic_raw() -> None:
    """Explore raw Anthropic API response structure."""
    print_separator("RAW ANTHROPIC RESPONSE STRUCTURE")

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    prompt = "Say love in 5 different languages."
    print(f"\nPrompt: {prompt}\n")

    print("Calling Anthropic API...")
    raw_response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=50,
        temperature=0.7,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    print("\n" + "=" * 60)
    print("\nRAW RESPONSE OBJECT\n")
    print(f"Type: {type(raw_response)}")
    print(f"\nFull object response:\n{raw_response}\n")
    print("=" * 60)

    print("\nUNPACKING STRUCTURE\n")
    # print(dir(raw_response))  # Print all attribute names

    # Step 1: Top-level fields
    top_fields = list(raw_response.model_dump().keys())
    print(f"Top-level fields: {top_fields}")

    # Step 2: Field types
    response_dict = raw_response.model_dump()
    print("\nField Types")
    for key, value in response_dict.items():
        print(f"  {key}: {type(value)}")

    print("\n" + "=" * 60)

    # Step 3: Dive into content
    print(f"\nDive into 'content':\n{raw_response.content}\n")
    print(f"content: {type(raw_response.content)} with {len(raw_response.content)} block(s)")

    if raw_response.content:
        first_block = raw_response.content[0]
        print(f"  content[0]: {type(first_block)}")
        print(f"  content[0] fields: {list(first_block.model_dump().keys())}")

        print(f"\n  Dive into content[0].text:")
        print(f"    type: {type(first_block.text)} = {first_block.text}")

    print("\n" + "=" * 60)

    # Step 4: Usage inspection
    print(f"\nDive into 'usage':\n{raw_response.usage}\n")
    print(f"usage: {type(raw_response.usage)}")
    print(f"usage fields: {list(raw_response.usage.model_dump().keys())}")

    for field, value in raw_response.usage.model_dump().items():
        print(f"  {field}: {type(value)}")

    print("\n" + "=" * 60)

    # Step 5: Simple fields
    print("\nSimple Fields")
    print(f"id: {type(raw_response.id)} = {raw_response.id}")
    print(f"model: {type(raw_response.model)} = {raw_response.model}")
    print(f"role: {type(raw_response.role)} = {raw_response.role}")
    print(f"stop_reason: {type(raw_response.stop_reason)} = {raw_response.stop_reason}")

    print("\n" + "=" * 60)
    print("\nAnthropic raw exploration complete!")


def explore_unsloth_raw() -> None:
    """Explore raw HuggingFace/Transformers response structure."""
    print_separator("RAW UNSLOTH/TRANSFORMERS RESPONSE STRUCTURE")

    model_name = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
    prompt = "Say love in 5 different languages."
    print(f"\nModel: {model_name}")
    print(f"\nPrompt: {prompt}")
    print("\n" + "=" * 60)

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"Using device: {device}")

    cache_dir = os.path.join('hf_cache', 'hub')

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        local_files_only=True
    )
    print(f"tokenizer type: {type(tokenizer)}")

    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        dtype=torch.float16,
        cache_dir=cache_dir,
        local_files_only=True
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    print(f"model type: {type(model)}")
    print("\n" + "=" * 60)

    print("\nUNPACKING STRUCTURE\n")

    # Step 1: Tokenizer encode path
    print("Step 1: TOKENIZER ENCODE")
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
        add_special_tokens=True,
    )
    print(f"inputs type: {type(inputs)}")
    print(f"inputs keys: {list(inputs.keys())}")

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print("\n[ENCODED TENSORS - CPU]")
    print(f"input_ids type: {type(input_ids)}")
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"input_ids dtype: {input_ids.dtype}")
    print(f"input_ids sample: {input_ids}")

    print(f"\nattention_mask type: {type(attention_mask)}")
    print(f"attention_mask shape: {tuple(attention_mask.shape)}")
    print(f"attention_mask dtype: {attention_mask.dtype}")
    print(f"attention_mask sample: {attention_mask}")

    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    print("\n[ENCODED TENSORS - ON DEVICE]")
    print(f"input_ids device: {inputs['input_ids'].device}")
    print(f"attention_mask device: {inputs['attention_mask'].device}")

    print("\n" + "=" * 60)

    # Step 2: Generation output tensor
    print("\nStep 2: MODEL.GENERATE (RAW OUTPUT)")
    with torch.no_grad():
        raw_output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False,  # keep tensor output (simple + consistent)
        )

    print("\n[RAW OUTPUT]")
    print(f"raw_output type: {type(raw_output)}")
    print(f"raw_output shape: {tuple(raw_output.shape)}")  # (B, S), e.g., (1, 59) -> 1 is batch size, 59 is the number of produced tokens
    print(f"raw_output dtype: {raw_output.dtype}")
    print(f"raw_output device: {raw_output.device}")
    print(f"raw_output sample (first 16 tokens): {raw_output[:, : min(16, raw_output.shape[1])]}")

    print("\n" + "=" * 60)

    # Step 3: Decoding (contains: prompt + generated_text)
    print("\nStep 3: TOKENIZER DECODE")
    decoded_full = tokenizer.decode(raw_output[0], skip_special_tokens=True)
    print(f"decoded_full type: {type(decoded_full)}")
    print(f"\n[DECODED FULL TEXT]\n{decoded_full}\n")

    # "Response only" (remove the prompt)
    response_only = decoded_full[len(prompt):].strip() if decoded_full.startswith(prompt) else decoded_full
    print("[RESPONSE ONLY]")
    print(response_only)

    print("\n" + "=" * 60)

    # Step 4: Token counts
    print("\nStep 4: TOKEN COUNTS")
    input_tokens = inputs["input_ids"].shape[1]
    total_tokens = raw_output.shape[1]
    new_tokens = total_tokens - input_tokens
    print(f"input_tokens: {input_tokens} ({type(input_tokens)})")
    print(f"total_tokens: {total_tokens} ({type(total_tokens)})")
    print(f"new_tokens: {new_tokens} ({type(new_tokens)})")

    print("\n" + "=" * 60)
    print("\nUnsloth raw exploration complete!")


def show_model_response_wrapping() -> None:
    """Show how raw responses are wrapped into ModelResponse."""

    print_separator("HOW RAW RESPONSES BECOME ModelResponse")

    print("\nOpenAI → ModelResponse\n")
    print("raw_response.choices[0].message.content    → ModelResponse.text")
    print("raw_response.usage.prompt_tokens           → ModelResponse.input_tokens")
    print("raw_response.usage.completion_tokens       → ModelResponse.output_tokens")
    print("raw_response.model                         → ModelResponse.model_name")
    print("raw_response.choices[0].message.tool_calls → ModelResponse.tool_calls")
    print("time.time() - start_time                   → ModelResponse.response_time")
    print("calculate_cost(...)                        → ModelResponse.cost")
    print("\nOpenAI raw exploration complete!")
    print("-" * 60)

    print("\nAnthropic → ModelResponse\n")
    print("''.join([b.text for b in raw_response.content if b.type=='text']) → ModelResponse.text")
    print("raw_response.usage.input_tokens                                   → ModelResponse.input_tokens")
    print("raw_response.usage.output_tokens                                  → ModelResponse.output_tokens")
    print("raw_response.model                                                → ModelResponse.model_name")
    print("[{id, name, input} for b in content if b.type=='tool_use']        → ModelResponse.tool_calls")
    print("time.time() - start_time                                          → ModelResponse.response_time")
    print("calculate_cost(...)                                               → ModelResponse.cost")
    print("\nAnthropic raw exploration complete!")
    print("-" * 60)

    print("\nUnsloth/HF → ModelResponse\n")
    print("tokenizer.decode(outputs[0])[len(prompt):].strip() → ModelResponse.text")
    print("inputs['input_ids'].shape[1]                          → ModelResponse.input_tokens")
    print("outputs.shape[1] - input_tokens                    → ModelResponse.output_tokens")
    print("model_name                                         → ModelResponse.model_name")
    print("[]                                                 → ModelResponse.tool_calls (empty)")
    print("time.time() - start_time                           → ModelResponse.response_time")
    print("0.0                                                → ModelResponse.cost (free)")
    print("\nUnsloth raw exploration complete!")
    print("-" * 60)

    print("\nThe Result: Unified Interface\n")
    print("All three providers return the SAME ModelResponse dataclass:")
    print("  @dataclass")
    print("  class ModelResponse:")
    print("      text: str")
    print("      input_tokens: int")
    print("      output_tokens: int")
    print("      model_name: str")
    print("      tool_calls: List[Dict[str, Any]]")
    print("      response_time: float")
    print("      cost: float")

    print("\nThis is why call_model() works identically for all providers!")
    print("\n" + "=" * 60)


def explore_all() -> None:
    """Run all explorations sequentially."""
    explore_openai_raw()
    explore_anthropic_raw()
    explore_unsloth_raw()
    show_model_response_wrapping()


if __name__ == "__main__":
    # explore_openai_raw()
    # explore_anthropic_raw()
    # explore_unsloth_raw()
    # show_model_response_wrapping()

    explore_all()

