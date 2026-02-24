"""Agent Exploration Script."""

import os
import json
from typing import Any

from src.utils import print_separator, capture_and_save
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


# Tool definitions for OpenAI function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "Search for hotels matching price range and area constraints",
            "parameters": {
                "type": "object",
                "properties": {
                    "pricerange": {
                        "type": "string",
                        "enum": ["cheap", "moderate", "expensive"],
                        "description": "Price category"
                    },
                    "area": {
                        "type": "string",
                        "enum": ["centre", "north", "south", "east", "west"],
                        "description": "Location area"
                    }
                },
                "required": ["pricerange", "area"]
            }
        }
    }
]


def create_dummy_scenario() -> dict[str, Any]:
    """
    Create a hardcoded hotel search scenario mimicking MultiWOZ structure.

    Purpose: Simulate a user query with slot constraints
    Returns: Dictionary with user query and expected slots
    """
    dummy_scenario = {
        "user_query": "I need a cheap hotel in the center",
        "expected_slots": {
            "domain": "hotel",
            "pricerange": "cheap",
            "area": "centre"
        },
        "database": [
            {"name": "Hotel A", "pricerange": "cheap", "area": "centre", "rating": 4.2},
            {"name": "Hotel B", "pricerange": "cheap", "area": "centre", "rating": 3.8},
            {"name": "Hotel C", "pricerange": "expensive", "area": "centre", "rating": 4.9}
        ]
    }

    print_separator("DUMMY SCENARIO")
    print(json.dumps(dummy_scenario, indent=2))

    return dummy_scenario


def search_hotels(pricerange: str, area: str, database: list[dict[str, Any]], verbose: bool = False) -> list[dict[str, Any]]:
    """
    Search hotel database by pricerange and area constraints.

    Purpose: Simulate a tool/API that agents can call
    Args:
        pricerange: Price category (cheap, moderate, expensive)
        area: Location area (center, north, south, east, west)
        database: List of hotel records to search
        verbose: If True, print execution details (default: False)
    Returns: List of matching hotels
    """
    search_results = []
    for hotel in database:
        if hotel.get("pricerange") == pricerange and hotel.get("area") == area:
            search_results.append(hotel)

    if verbose:
        print_separator("SEARCH TOOL EXECUTION")
        print(f"Query: pricerange='{pricerange}', area='{area}'")
        print(f"\nFound {len(search_results)} matching hotels:")
        for hotel in search_results:
            print(f"  - {hotel['name']} (rating: {hotel['rating']})")

    return search_results


def call_llm(system_prompt: str = "", user_query: str = "", model_name: str = "gpt-4o-mini", tools: list[dict[str, Any]] = None, temperature: float = 0, messages: list[dict[str, Any]] = None) -> Any:
    """
    Helper function to call OpenAI LLM with consistent settings.

    Purpose: Centralize LLM API calls to avoid code duplication.
             For simplicity, we use only gpt-4o-mini in these demonstrations (our goal is to understand agent architecture, not compare model quality).

    Args:
        system_prompt: System instructions (ignored if messages provided)
        user_query: User's input (ignored if messages provided)
        model_name: Which OpenAI model to use (default: gpt-4o-mini)
        tools: Optional tool definitions for function calling
        temperature: Sampling temperature (0 = deterministic)
        messages: Optional pre-built conversation history (overrides system_prompt/user_query)
    Returns: OpenAI ChatCompletion response object

    Usage:
        Simple call: call_llm(system_prompt="...", user_query="...")
        Complex call: call_llm(messages=[...], model_name="...")
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Use messages if provided, otherwise build from system_prompt/user_query
    if messages is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

    # Build request params
    request_params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature
    }

    # Only add tools if provided
    if tools is not None:
        request_params["tools"] = tools

    response = client.chat.completions.create(**request_params)

    return response


def run_standard_llm(user_query: str, model_name: str = "gpt-4o-mini") -> dict[str, Any]:
    """
    Raw LLM call without any tool definitions or reasoning framework (Zero-Shot).
    Purpose: Pure baseline - model answers from training data only.

    Args:
        user_query: The user's question/request
        model_name: Which LLM to use
    Returns: Dictionary with model response and metadata
    """
    system_prompt = """You are a helpful hotel search assistant.
Help the user find hotels based on their preferences."""

    print_separator(f"STANDARD LLM - NO TOOLS ({model_name})")
    print(f"User Query: {user_query}")

    response = call_llm(
        system_prompt=system_prompt,
        user_query=user_query,
        model_name=model_name
    )

    answer = response.choices[0].message.content

    print(f"\nLLM Response:\n{answer}")

    result = {
        "agent_type": "standard_llm",
        "model": model_name,
        "user_query": user_query,
        "tools_available": [],
        "tools_used": [],
        "response": answer,
        "tokens_used": response.usage.total_tokens
    }

    return result


def run_simple_agent(user_query: str, database: list[dict[str, Any]], model_name: str = "gpt-4o-mini") -> dict[str, Any]:
    """
    Agent with tool calling: single request-response cycle with tool execution.
    Purpose: Model calls tools once, then generates final answer with results

    Args:
        user_query: The user's question/request
        database: Hotel database to search
        model_name: Which LLM to use
    Returns: Dictionary with model response and actual tool usage
    """
    system_prompt = """You are a helpful hotel search assistant.
You have access to a search_hotels tool if needed to find hotels matching specific criteria."""

    print_separator(f"SIMPLE AGENT - WITH TOOL CALLING ({model_name})")
    print(f"User Query: {user_query}")

    # Step 1: Initial LLM call
    response = call_llm(
        system_prompt=system_prompt,
        user_query=user_query,
        model_name=model_name,
        tools=TOOL_DEFINITIONS
    )

    message = response.choices[0].message
    tools_used = []
    tool_results = []

    print_separator("STEP 1: LLM DECISION (First LLM Call)")

    # Step 2: Check if model wants to call a tool
    if message.tool_calls:
        print(f"Decision: Use tool(s) to gather information")
        print(f"\nModel requested {len(message.tool_calls)} tool call(s):")

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"  - {tool_name}({tool_args})")
            tools_used.append(tool_name)

            # Step 3: Execute the tool
            if tool_name == "search_hotels":
                simple_results = search_hotels(
                    pricerange=tool_args["pricerange"],
                    area=tool_args["area"],
                    database=database,
                    verbose=True  # Enable printing
                )
                tool_results.append(simple_results)

                print_separator("STEP 2: TOOL EXECUTION RESULTS")
                print(f"Retrieved {len(simple_results)} hotels:")
                print(json.dumps(simple_results, indent=2))

        # Step 4: Send tool results back to model for final answer
        print_separator("STEP 3: GENERATING FINAL RESPONSE WITH TOOL RESULTS (Second LLM Call)")

        # Build conversation: user → assistant tool call → tool result
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
            message,  # Assistant's decision to call tool
            {
                "role": "tool",
                "tool_call_id": message.tool_calls[0].id,
                "content": json.dumps(tool_results[0])
            }
        ]

        # Step 5: Get final answer from model
        final_response = call_llm(
            model_name=model_name,
            messages=messages
        )

        final_answer = final_response.choices[0].message.content
        total_tokens = response.usage.total_tokens + final_response.usage.total_tokens

        print(f"Final Answer:\n{final_answer}")

    else:
        # No tool call - use direct response
        print("Decision: Answer directly without tools")
        final_answer = message.content
        total_tokens = response.usage.total_tokens

    print_separator("FINAL ANSWER")
    print(final_answer)

    result = {
        "agent_type": "simple_agent",
        "model": model_name,
        "user_query": user_query,
        "tools_available": ["search_hotels"],
        "tools_used": tools_used,
        "tool_results": tool_results,
        "response": final_answer,
        "tokens_used": total_tokens
    }

    return result


def run_react_agent(user_query: str, database: list[dict[str, Any]], model_name: str = "gpt-4o-mini", max_iterations: int = 5) -> dict[str, Any]:
    """
    ReAct-style agent with iterative Thought-Action-Observation loop.
    Purpose: Agent can reason, act, observe results, and repeat until task complete.

    Args:
        user_query: The user's question/request
        database: Hotel database to search
        model_name: Which LLM to use
        max_iterations: Maximum reasoning loops to prevent infinite cycles
    Returns: Dictionary with complete reasoning trace and final response
    """
    system_prompt = """You are a helpful hotel search assistant using ReAct reasoning.
You have access to a search_hotels tool if needed.

Follow this pattern for each step:
Thought: [Reason about what you need to do]
Action: [Either use a tool or provide Final Answer]

Available actions:
- search_hotels(pricerange, area) - search for hotels
- Final Answer: [your response to the user]

Continue reasoning until you can provide a Final Answer."""

    print_separator(f"REACT AGENT - ITERATIVE REASONING ({model_name})")
    print(f"User Query: {user_query}")
    print(f"Max Iterations: {max_iterations}")

    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    tools_used = []
    tool_results = []
    reasoning_trace = []

    for iteration in range(1, max_iterations + 1):
        print_separator(f"ITERATION {iteration}")


        # Step 1 (THINK): Agent thinks and decides action
        response = call_llm(
            model_name=model_name,
            messages=conversation_history,
            tools=TOOL_DEFINITIONS
        )

        message = response.choices[0].message

        # Step 2 (ACT): Execute tool, if decided

        # Check if agent wants to use a tool
        if message.tool_calls:
            print("Action: Use tool")

            # Track this iteration's actions
            iteration_tools = []
            iteration_observations = []

            # First, add the assistant's message with all tool calls
            conversation_history.append(message)

            # Then process each tool call and add results
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"  → {tool_name}({tool_args})")
                tools_used.append(tool_name)

                if tool_name == "search_hotels":
                    react_results = search_hotels(
                        pricerange=tool_args["pricerange"],
                        area=tool_args["area"],
                        database=database
                    )
                    tool_results.append(react_results)

                    print(f"Observation: Found {len(react_results)} hotels")

                    # Add this specific tool result to conversation
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(react_results)
                    })

                    # Track for this iteration
                    iteration_tools.append({
                        "tool": tool_name,
                        "args": tool_args
                    })
                    iteration_observations.append(react_results)

            # Add one trace entry per iteration (not per tool call)
            reasoning_trace.append({
                "iteration": iteration,
                "action": "tool_calls",
                "tools": iteration_tools,
                "observations": iteration_observations
            })
        else:
            # Agent provided final answer
            print("Action: Provide Final Answer")
            final_answer = message.content
            print(f"\nFinal Answer:\n{final_answer}")

            reasoning_trace.append({
                "iteration": iteration,
                "action": "final_answer",
                "response": final_answer
            })

            break
    else:
        # Max iterations reached without final answer
        final_answer = "Maximum iterations reached without completing task."
        print(f"\n{final_answer}")

    print_separator("REACT AGENT SUMMARY")
    print(f"Total Iterations: {len(reasoning_trace)}")
    print(f"  - Iteration = one complete LLM reasoning cycle")
    print(f"LLM Calls Made: {len(reasoning_trace)}")
    print(f"Tools Executed: {len(tools_used)} calls")
    print(f"Tool Breakdown: {', '.join(tools_used)}")

    result = {
        "agent_type": "react",
        "model": model_name,
        "user_query": user_query,
        "tools_available": ["search_hotels"],
        "tools_used": tools_used,
        "tool_results": tool_results,
        "reasoning_trace": reasoning_trace,
        "iterations_used": len(reasoning_trace),
        "response": final_answer,
        "tokens_used": "N/A"  # No need to sum across all calls at this phase
    }

    return result


def main() -> None:
    """Explore and compare three agent architectures: Standard LLM, Simple Agent, ReAct Agent."""

    # Create the dummy scenario
    scenario = create_dummy_scenario()

    print_separator("TEST BASIC AGENT EXPLORATION")
    print("\nTesting 3 agent types: Standard LLM, Simple Agent, ReAct Agent")

    # Test the search tool
    search_hotels(pricerange="cheap", area="centre", database=scenario["database"])

    # Test a simple llm
    run_standard_llm(user_query=scenario["user_query"])

    # Test simple agent with relevant query
    run_simple_agent(user_query=scenario["user_query"], database=scenario["database"])

    # Test simple agent with irrelevant query (should NOT use tool)
    run_simple_agent(user_query="What is your cancellation policy?", database=scenario["database"])

    # Test ReAct agent with a simple query (2 iterations expected)
    run_react_agent(user_query=scenario["user_query"], database=scenario["database"])

    # Test ReAct agent with another (complex) multi-step query
    run_react_agent(user_query="I want a cheap hotel in the centre, but also show me expensive hotels in the same area for comparison.",
                    database=scenario["database"])

    print_separator("END OF TEST AGENT EXPLORATION")


if __name__ == "__main__":
    capture_and_save(func=main, output_path="docs/agents_inspection/agent_exploration.txt")

