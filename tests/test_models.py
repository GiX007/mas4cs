"""Open-Source and API-based LLM call tests."""

import sys
import json
from typing import Any
from pathlib import Path
from tabulate import tabulate
from datetime import datetime

from src.models import call_model, UNSLOTH_MODELS, PAID_MODELS, ModelResponse
from src.utils import print_separator


def test_gpu_availability() -> None:
    """
    Check if GPU is available for model inference.
    Reports PyTorch version, CUDA availability, and GPU specifications.

    Note:
        GPU is recommended but not required. Models will run on CPU if no GPU detected.
    """
    try:
        import torch
    except ImportError:
        print("\n PyTorch not installed. Run: pip install torch")
        return

    print_separator("GPU AVAILABILITY CHECK")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("CUDA device count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print("GPU device:", torch.cuda.current_device())
        print(f"GPU device name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print("\nGPU detected - models will run efficiently")
    else:
        print("\n No GPU detected - models will use CPU (10-20x slower)")
        print("  For faster inference, install CUDA-enabled PyTorch:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu118")


def test_model_registry() -> None:
    """
    Display model registry statistics.
    Shows total count, free vs paid breakdown, and complete lists.
    """
    total = len(UNSLOTH_MODELS) + len(PAID_MODELS)

    print_separator("MODEL REGISTRY SUMMARY")
    print(f"Total models available: {total}")
    print(f"  - Free models (Unsloth): {len(UNSLOTH_MODELS)}")
    print(f"  - Paid models (APIs): {len(PAID_MODELS)}")
    print()

    print("OPEN-SOURCE MODELS (Unsloth/HuggingFace):")
    for i, model in enumerate(UNSLOTH_MODELS, 1):
        print(f"  {i}. {model}")
    print()

    print("PAID MODELS (OpenAI/Anthropic):")
    for i, model in enumerate(PAID_MODELS, 1):
        print(f"  {i}. {model}")
    print()


def test_openai_call() -> None:
    """
    Test real OpenAI API call.
    Verifies connection, response format, and token counting.

    Requires:
        - OPENAI_API_KEY in .env file
        - openai package installed
    """
    print_separator("TESTING OPENAI API")

    try:
        response = call_model(
            model_name="gpt-4o-mini",
            prompt="Say 'Hello from OpenAI!' and nothing else.",
            max_tokens=20,
            temperature=0.0
        )

        print(f"Response: {response.text}")
        print(f"Input tokens: {response.input_tokens}")
        print(f"Output tokens: {response.output_tokens}")
        print(f"Model: {response.model_name}")
        print(f"Response time: {response.response_time:.2f}s")
        print(f"Cost: ${response.cost:.6f}")

        # Verify it's a real response
        assert "Hello from OpenAI" in response.text, "Response doesn't contain expected text"
        assert response.input_tokens > 0, "Input tokens should be > 0"
        assert response.output_tokens > 0, "Output tokens should be > 0"
        assert response.response_time > 0, "Response time should be > 0"
        assert response.cost > 0, "Cost should be > 0 for paid model"

        print("\nOpenAI test passed!")

    except Exception as e:
        print(f"\nOpenAI test failed: {e}")


def test_anthropic_call() -> None:
    """
    Test real Anthropic (Claude) API call.
    Verifies connection, response format, and token counting.

    Requires:
        - ANTHROPIC_API_KEY in .env file
        - anthropic package installed
    """
    print_separator("TESTING ANTHROPIC API")

    try:
        response = call_model(
            model_name="claude-3-haiku-20240307",
            prompt="Say 'Hello from Claude!' and nothing else.",
            max_tokens=20,
            temperature=0.0
        )

        print(f"Response: {response.text}")
        print(f"Input tokens: {response.input_tokens}")
        print(f"Output tokens: {response.output_tokens}")
        print(f"Model: {response.model_name}")
        print(f"Response time: {response.response_time:.2f}s")
        print(f"Cost: ${response.cost:.6f}")

        # Verify it's a real response
        assert "Hello from Claude" in response.text, "Response doesn't contain expected text"
        assert response.input_tokens > 0, "Input tokens should be > 0"
        assert response.output_tokens > 0, "Output tokens should be > 0"
        assert response.response_time > 0, "Response time should be > 0"
        assert response.cost > 0, "Cost should be > 0 for paid model"

        print("\nAnthropic test passed!")

    except Exception as e:
        print(f"\nAnthropic test failed: {e}")


def test_unsloth_call() -> None:
    """
    Test Unsloth/HuggingFace model loading and inference.
    Uses the smallest model (3B) for faster testing.

    Warning:
        First run downloads ~2GB model (cached for future runs).
        Use GPU, CPU is too slow.
    """
    print_separator("TESTING UNSLOTH MODEL (LOCAL)")
    print("Note: First run will download model (~2GB)")
    print("This may take some minutes...\n")

    try:
        # Use the smallest model for testing
        response = call_model(
            model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
            prompt="Say 'Hello from Llama!' and nothing else.",
            max_tokens=20,
            temperature=0.0
        )

        print(f"\nResponse: {response.text}")
        print(f"Input tokens: {response.input_tokens}")
        print(f"Output tokens: {response.output_tokens}")
        print(f"Model: {response.model_name}")
        print(f"Response time: {response.response_time:.2f}s")
        print(f"Cost: ${response.cost:.6f} (FREE)")

        # Verify it's a real response
        assert len(response.text) > 0, "Response should not be empty"
        assert response.input_tokens > 0, "Input tokens should be > 0"
        assert response.output_tokens > 0, "Output tokens should be > 0"
        assert response.response_time > 0, "Response time should be > 0"
        assert response.cost == 0.0, "Cost should be 0.0 for free model"

        print("\nUnsloth test passed!")

    except Exception as e:
        print(f"\nUnsloth test failed: {e}")


def test_offline_models() -> None:
    """
    Test that cached models work without internet connection.
    Run this after disconnecting from internet to verify local cache.
    """
    print_separator("OFFLINE MODEL TEST (No Internet Required)")
    print("Testing cached models from hf_cache/hub/\n")

    models_to_test =[
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
        # "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    ]

    prompt = "Say hallo in 5 different languages"
    print(f"Prompt: {prompt}\n")

    all_passed = True

    for model_name in models_to_test:
        model_short = model_name.split('/')[-1]
        print(f"[{model_short}]")

        try:
            response = call_model(model_name=model_name, prompt=prompt, max_tokens=100, temperature=0.0)

            print(f"Response: {response.text}")
            print(f"\nLatency: {response.response_time:.2f}s\n")
        except Exception as e:
            print(f"FAILED: {e}\n")
            all_passed = False

    if all_passed:
        print("All tests passed -> models are cached and work successfully offline!")
    else:
        print("Some tests failed -> check errors above")


def save_test_results(results: list[dict[str, Any]], output_path: str) -> None:
    """
    Save test results to timestamped JSON file.

    Args:
        results: List of test result dictionaries
        output_path: Full path where to save (e.g., "docs/model_inspection/model_responses")

    Example:
        save_test_results(results, "docs/model_inspection/model_responses")
        # Saves to: docs/model_inspection/model_responses_20250210_143022.json
    """
    # Parse path
    path = Path(output_path)  # Convert string to Path object
    save_dir = path.parent  # → Path("docs/model_inspection")
    base_name = path.name  # → "model_responses"

    # Create directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = save_dir / f"{base_name}_{timestamp}.json"

    # Prepare output with metadata
    output = {
        "test_name": base_name,
        "timestamp": timestamp,
        "num_results": len(results),
        "results": results
    }

    # Save as JSON
    Path(filename).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f"\nResults saved to: {filename}")


def test_model_responses() -> None:
    """
    Test multiple models on a qualitative prompt.
    Measures response quality, token usage, and latency.

    Compares:
    - Paid models (GPT-4o-mini, Claude Haiku)
    - Open-source models
    """
    # Models to test
    test_models: list[str] = [

        # API-based models
        "gpt-4o-mini",
        "claude-3-haiku-20240307",

        # Small Open-Source models
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-3B-Instruct-bnb-4bit",

        # Medium Open-Source models
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "unsloth/gemma-2-9b-it-bnb-4bit",

        # Large Open-Source models
        "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",

        # Unknown model
        "fake-model-123"
    ]

    # Qualitative prompt to evaluate reasoning
    prompt = (
        "A customer wants to book a hotel for 3 nights but didn't specify the city. "
        "What should a customer service agent ask first?"
        "Return ONLY a valid JSON with keys: next_question, reason, required_slots (list). No other keys. No markdown. No preamble. No explanation. No verbosity."
    )

    print_separator("MODEL COMPARISON TEST")
    print(f"\nPrompt: {prompt}")
    print("\n" + "=" * 60)

    results: list[dict] = []

    for model_name in test_models:
        print(f"\n[Testing: {model_name}]")

        try:
            response: ModelResponse = call_model(
                model_name=model_name,
                prompt=prompt,
                max_tokens=100,  # Short response for quick comparison
                temperature=0.7
            )

            # Store results
            results.append({
                "model": model_name,
                "response": response.text,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "response_time": response.response_time,
                "cost": response.cost,
                "tool_calls": len(response.tool_calls)
            })

            # Display results
            print(f"\nResponse: {response.text}")
            print(f"\nTokens (in/out): {response.input_tokens}/{response.output_tokens}")
            print(f"Latency: {response.response_time:.2f}s")
            print(f"Tool calls: {len(response.tool_calls)}")
            print("\n" + "=" * 60)

        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "model": model_name,
                "response": "ERROR",
                "error": str(e)
            })

    # Save results
    save_test_results(results, "docs/model_inspection/model_responses")

    # Summary comparison
    print_separator("SUMMARY COMPARISON")

    # Performance metrics table
    table_data = []
    for r in results:
        if "error" not in r:
            model_short = r['model'].split('/')[-1] if '/' in r['model'] else r['model']
            cost_str = f"${r['cost']:.6f}" if r['cost'] > 0 else "FREE"
            table_data.append([model_short, f"{r['response_time']:.2f}s", r['output_tokens'], cost_str])

    summary_table = tabulate(
        table_data,
        headers=["Model", "Latency", "Tokens", "Cost"],
        tablefmt="github"  # or "simple", "grid", "fancy_grid"
    )
    print(summary_table)

    # Response quality comparison
    print_separator("RESPONSE QUALITY COMPARISON")

    quality_comparison = []
    for i, r in enumerate(results, 1):
        if "error" not in r:
            model_short = r['model'].split('/')[-1] if '/' in r['model'] else r['model']
            response_text = f"{r['response']}..." if len(r['response']) > 300 else r['response']
            quality_comparison.append(f"[{i}] {model_short}:\n{response_text}")
            print(f"\n[{i}] {model_short}:")
            print(f"{response_text}")
            print("-" * 60)

    print_separator("MODEL RESPONSES TEST COMPLETE")

    # Save Summary to .txt
    summary_dir = Path("docs/model_inspection")
    summary_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = summary_dir / f"model_responses_summary_{timestamp}.txt"

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("MODEL RESPONSES TEST SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Models tested: {len(results)}\n\n")

        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 60 + "\n")
        f.write(summary_table + "\n\n")

        f.write("RESPONSE QUALITY COMPARISON\n")
        f.write("-" * 60 + "\n")
        for comp in quality_comparison:
            f.write(comp + "\n\n")

        f.write("=" * 60 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 60 + "\n")

    print(f"\nResults saved to: docs/model_inspection/model_responses_{timestamp}.json")
    print(f"Summary saved to: {summary_file}")


def test_reasoning_capability() -> None:
    """Test models on prompts that reveal reasoning differences."""
    from tests.test_prompts import TEST_CASES

    # Models to compare
    models_to_test = [
        "gpt-4o-mini",
        "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    ]

    # Prepare summary file
    summary_dir = Path("docs/model_inspection")
    summary_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = summary_dir / f"reasoning_capability_summary_{timestamp}.txt"

    # Open file for writing throughout test
    with open(summary_file, 'w', encoding='utf-8') as f:

        print_separator("REASONING CAPABILITY TEST")
        header = f"\nTesting {len(TEST_CASES)} scenarios across {len(models_to_test)} models\n"
        print(header)
        f.write(header)

        all_results = []

        for i, test_case in enumerate(TEST_CASES, 1):
            test_header = "=" * 60 + f"\nTEST {i}/{len(TEST_CASES)}: {test_case['name']}\n" + "=" * 60 + f"\nPrompt:\n{test_case['prompt']}\n\n" + "-" * 60
            print(test_header)
            f.write("\n" + test_header + "\n")

            test_result = {
                "test_name": test_case['name'],
                "prompt": test_case['prompt'],
                "responses": []
            }

            for model_name in models_to_test:
                model_short = model_name.split('/')[-1] if '/' in model_name else model_name
                model_line = f"\n[{model_short}]"
                print(model_line)
                f.write(model_line + "\n")

                try:
                    response = call_model(model_name=model_name, prompt=test_case['prompt'], max_tokens=200, temperature=0.0)

                    # Store result
                    test_result["responses"].append({
                        "model": model_name,
                        "response": response.text,
                        "latency": response.response_time,
                        "tokens": response.output_tokens,
                        "cost": response.cost
                    })

                    response_text = f"Response:\n{response.text}\nLatency: {response.response_time:.2f}s | Tokens: {response.output_tokens}"
                    print(response_text)
                    f.write(response_text + "\n")

                except Exception as e:
                    error_text = f"ERROR: {e}"
                    print(error_text)
                    f.write(error_text + "\n")
                    test_result["responses"].append({
                        "model": model_name,
                        "error": str(e)
                    })

                separator = "\n" + "=" * 60
                print(separator)
                f.write(separator + "\n")

            all_results.append(test_result)
            print()
            f.write("\n")

        print_separator("REASONING TEST COMPLETE")
        f.write("\n")

    # Save JSON results
    save_test_results(all_results, "docs/model_inspection/reasoning_capability")

    print(f"Summary saved to: {summary_file}")


def run_tests(test_keys: list[str]) -> None:
    """
    Runs selected test functions by name.

    Args: test_keys - list of test name strings
    Return: None

    CLI usage:
        python -m tests.test_models gpu
        python -m tests.test_models gpu model_registry openai
        python -m tests.test_models  # runs defaults: responses, reasoning

    Programmatic usage:
        run_tests(["gpu", "openai"])
    """
    test_map = {
        "gpu": test_gpu_availability,
        "model_registry": test_model_registry,
        "openai": test_openai_call,
        "anthropic": test_anthropic_call,
        "unsloth": test_unsloth_call,
        "offline": test_offline_models,
        "responses": test_model_responses,
        "reasoning": test_reasoning_capability,
    }

    for key in test_keys:
        if key in test_map:
            test_map[key]()
        else:
            print(f"Unknown test '{key}'. Choose from: {list(test_map.keys())}")


if __name__ == "__main__":

    keys = sys.argv[1:] if len(sys.argv) > 1 else ["responses", "reasoning"]
    run_tests(keys)

