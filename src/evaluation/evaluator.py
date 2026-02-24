"""
Dialogue evaluation with hierarchical metrics.

Evaluates at three levels:
- Turn-level (micro): Per-turn metrics
- Dialogue-level (meso): Aggregated per dialogue
- Dataset-level (macro): Aggregated across all dialogues
"""

from typing import Any, Callable
from src.evaluation import (
    calculate_intent_accuracy, calculate_action_type_accuracy, calculate_slot_accuracy, calculate_jga, calculate_hallucination_rate,
    calculate_policy_compliance, calculate_task_success, calculate_system_correctness, calculate_domain_accuracy, calculate_memory_transfer_accuracy
)
from src.evaluation import create_judge_prompt, parse_judge_response


class DialogueEvaluator:
    """
    Comprehensive dialogue evaluation with hierarchical metrics.

    Evaluates at three levels:
    - Turn-level (micro): Per-turn metrics
    - Dialogue-level (meso): Aggregated per dialogue
    - Dataset-level (macro): Aggregated across all dialogues
    """

    def __init__(self, policy_requirements: dict[str, list[str]], judge_llm_fn: Callable[[str], str] | None = None) -> None:
        """
        Initialize evaluator with policy rules and optional LLM judge.

        Args:
            policy_requirements: Required slots per action
            judge_llm_fn: Function to call LLM judge (takes prompt string, returns response string)
        """
        self.policy_requirements = policy_requirements
        self.judge_llm_fn = judge_llm_fn
        self.turn_metrics = []
        self.dialogue_history = []

    def reset(self) -> None:
        """Clear all accumulated metrics for new dialogue."""
        self.turn_metrics = []
        self.dialogue_history = []

    def print_turn(self) -> None:
        """Prints the metrics of the turn."""
        print(self.turn_metrics)

    def evaluate_turn(
            self,
            turn_id: int,
            predicted_slots: dict[str, dict[str, str]],
            ground_truth_slots: dict[str, dict[str, str]],
            predicted_intent: str,
            ground_truth_intent: str,
            predicted_act_type: list[str],
            ground_truth_act_type: list[str],
            predicted_domain: str,
            action_taken: str,
            user_message: str | None = None,
            system_response: str | None = None
    ) -> dict[str, Any]:
        """
        Evaluate a single dialogue turn with all metrics.

        Args:
            turn_id: Turn number
            predicted_slots: System's slot tracking
            ground_truth_slots: Annotated ground truth
            predicted_intent: System's detected intent
            ground_truth_intent: Annotated intent
            predicted_act_type: System's dialogue acts
            ground_truth_act_type: Annotated acts
            predicted_domain: System's routed domain
            action_taken: Action the system took
            user_message: Optional user message for LLM judge
            system_response: Optional system response for LLM judge

        Returns:
            Dictionary with all turn metrics
        """
        # Calculate (objective) metrics

        # 1. Domain accuracy (predicted_domain = state["current_domain"], gt extracted from active_intent via split("_")[-1] from user's turn)
        domain_acc, domain_correct, gt_domain = calculate_domain_accuracy(predicted_domain, ground_truth_intent)

        # 2. Intent accuracy (predicted_intent = state["active_intent"] (format: "{intent}_{domain}"), gt extracted from frame["active_intent"] from user's turn)
        intent_acc, intent_correct = calculate_intent_accuracy(predicted_intent, ground_truth_intent)

        # 3. Action-type accuracy
        act_acc, act_correct = calculate_action_type_accuracy(predicted_act_type, ground_truth_act_type)
        # print(f"\n    act: acc={act_acc:.2f} | predicted={predicted_act_type} | gt={ground_truth_act_type}")

        # 4. Slot accuracy
        slot_acc, slot_correct, slot_total = calculate_slot_accuracy(predicted_slots, ground_truth_slots)

        # 5. JGA (Joint Goal Accuracy)
        jga, jga_breakdown = calculate_jga(predicted_slots, ground_truth_slots)

        # 6. Hallucination rate
        hall_rate, hall_count, predicted_count = calculate_hallucination_rate(predicted_slots, ground_truth_slots, current_domain=predicted_domain)
        # print(f"    hall: rate={hall_rate:.2f} count={hall_count} predicted={predicted_count}")

        # 7. Policy compliance (action_taken = result["intent"] if booking else result["action_type"], normalized to match BOOKING_REQUIRED_SLOTS keys e.g. "book_hotel")
        policy_ok, policy_reason = calculate_policy_compliance(action_taken, self.policy_requirements, predicted_slots)

        # 8. System Correctness
        system_correct, system_reason = calculate_system_correctness(
            predicted_action=action_taken,
            predicted_intent=predicted_intent,
            predicted_slots=predicted_slots,  # accumulated
            hallucination_detected=(hall_rate > 0),
            policy_compliant=policy_ok,
            current_domain=predicted_domain
        )
        # print(f"\n    sys: correct={system_correct} | reason={system_reason} | hall={hall_rate > 0} | policy={policy_ok} | action={action_taken} | intent={predicted_intent}")

        turn_result = {
            "turn_id": turn_id,
            "domain": predicted_domain,
            "predicted_slots": predicted_slots,
            "user_message": user_message or "",
            "system_response": system_response or "",
            "predicted_intent": predicted_intent,
            "ground_truth_intent": ground_truth_intent,

            # 1. Domain
            "domain_accuracy": domain_acc,
            "domain_correct": domain_correct,
            # 2. Intent
            "intent_accuracy": intent_acc,
            "intent_correct": intent_correct,
            # 3. Action type
            "action_type_accuracy": act_acc,
            "action_type_correct": act_correct,
            # 4. Slot
            "slot_accuracy": slot_acc,
            "slot_correct": slot_correct,
            "slot_total": slot_total,
            # 5. JGA
            "jga": jga,
            "jga_breakdown": jga_breakdown,
            # 6. Hallucination
            "hallucination_rate": hall_rate,
            "hallucination_count": hall_count,
            "prediction_count": predicted_count,
            # 7. Policy
            "policy_compliant": policy_ok,
            "policy_reason": policy_reason,
            "action": action_taken,
            # 8. System correctness
            "system_correct": system_correct,
            "system_reason": system_reason,
        }

        # LLM judge evaluation
        if self.judge_llm_fn and user_message and system_response:
            judge_prompt = create_judge_prompt(
                user_message,
                system_response,
                ground_truth_slots,
                list(self.policy_requirements.keys())
            )
            judge_response = self.judge_llm_fn(judge_prompt)
            # print(f"\n    turn_id: {turn_id} | judge raw: {judge_response[:100]}")
            judge_result = parse_judge_response(judge_response)
            # print(f"    judge parsed: score={judge_result.get('score')} error={judge_result.get('error')}")
            turn_result["judge_score"] = judge_result.get("score", 0)
            turn_result["judge_feedback"] = judge_result

        self.turn_metrics.append(turn_result)
        self.dialogue_history.append(turn_result)

        return turn_result


    def evaluate_dialogue(self, ground_truth_goal: dict[str, Any]) -> dict[str, Any]:
        """
        Evaluate complete dialogue with aggregated metrics.

        Args:
            ground_truth_goal: User's goal from annotations

        Returns:
            Dictionary with dialogue-level metrics
        """
        num_turns = len(self.turn_metrics)
        if num_turns == 0:
            return {
                "task_success": False,
                "task_reason": "No turns evaluated",
                "num_turns": 0
            }

        # Count policy violations
        policy_violations = sum(1 for t in self.turn_metrics if not t["policy_compliant"])

        # 9. Task success rate - requires booking completion + all required slots filled + no policy violations
        if policy_violations > 0:
            task_ok = False
            task_reason = f"Policy violations detected ({policy_violations} violations)"
        else:
            # print(f"\n  task: requires_booking={ground_truth_goal.get('requires_booking')} | domains={ground_truth_goal.get('domains')} | final_slots={self.turn_metrics[-1].get('predicted_slots')}")
            task_ok, task_reason = calculate_task_success(self.turn_metrics, ground_truth_goal)

        # 10. Memory transfer accuracy
        memory_acc, memory_correct, memory_total, memory_events = calculate_memory_transfer_accuracy(self.dialogue_history)
        # print(f"\n  memory: acc={memory_acc:.2f} correct={memory_correct}/{memory_total} events={memory_events}")

        # Calculate averages
        avg_domain_acc = sum(t["domain_accuracy"] for t in self.turn_metrics) / num_turns
        avg_intent_acc = sum(t["intent_accuracy"] for t in self.turn_metrics) / num_turns
        avg_act_acc = sum(t["action_type_accuracy"] for t in self.turn_metrics) / num_turns
        avg_slot_acc = sum(t["slot_accuracy"] for t in self.turn_metrics) / num_turns
        avg_jga = sum(t["jga"] for t in self.turn_metrics) / num_turns
        avg_hall_rate = sum(t["hallucination_rate"] for t in self.turn_metrics) / num_turns
        avg_system_correctness = sum(t["system_correct"] for t in self.turn_metrics) / num_turns

        # Calculate judge average if available
        judge_scores = [t.get("judge_score", 0) for t in self.turn_metrics if "judge_score" in t]
        avg_judge_score = sum(judge_scores) / len(judge_scores) if judge_scores else None

        return {
            # Task completion
            "task_success": task_ok,
            "task_reason": task_reason,
            "num_turns": num_turns,

            # Intent and routing
            "avg_intent_accuracy": avg_intent_acc,
            "avg_action_type_accuracy": avg_act_acc,
            "avg_domain_accuracy": avg_domain_acc,

            # Slot tracking
            "avg_jga": avg_jga,
            "avg_slot_accuracy": avg_slot_acc,
            "avg_hallucination_rate": avg_hall_rate,

            # System behavior
            "avg_system_correctness": avg_system_correctness,

            # Memory transfer
            "memory_transfer_accuracy": memory_acc,
            "memory_correct": memory_correct,
            "memory_total": memory_total,
            "memory_events": memory_events,

            # Policy
            "policy_violations": policy_violations,

            # LLM judge
            "avg_judge_score": avg_judge_score,

            # Detailed turn data
            "turn_metrics": self.turn_metrics
        }


class DatasetEvaluator:
    """Aggregate evaluation results across multiple dialogues."""

    def __init__(self) -> None:
        """Initialize dataset-level aggregator."""
        self.dialogue_results = []

    def add_dialogue(self, dialogue_result: dict[str, Any]) -> None:
        """Add a dialogue evaluation result."""
        self.dialogue_results.append(dialogue_result)

    def compute_dataset_metrics(self) -> dict[str, Any]:
        """
        Compute dataset-level (macro) metrics.

        Returns:
            Dictionary with aggregated metrics across all dialogues
        """
        num_dialogues = len(self.dialogue_results)
        if num_dialogues == 0:
            return {"num_dialogues": 0}

        # Task success rate
        task_success_rate = sum(d["task_success"] for d in self.dialogue_results) / num_dialogues

        # Average metrics across dialogues
        avg_domain_acc = sum(d["avg_domain_accuracy"] for d in self.dialogue_results) / num_dialogues
        avg_intent_acc = sum(d["avg_intent_accuracy"] for d in self.dialogue_results) / num_dialogues
        avg_act_acc = sum(d["avg_action_type_accuracy"] for d in self.dialogue_results) / num_dialogues
        avg_jga = sum(d["avg_jga"] for d in self.dialogue_results) / num_dialogues
        avg_slot_acc = sum(d["avg_slot_accuracy"] for d in self.dialogue_results) / num_dialogues
        avg_hall_rate = sum(d["avg_hallucination_rate"] for d in self.dialogue_results) / num_dialogues
        avg_system_correctness = sum(d["avg_system_correctness"] for d in self.dialogue_results) / num_dialogues

        # Memory transfer (only dialogues with transfers)
        memory_dialogues = [d for d in self.dialogue_results if d.get("memory_total", 0) > 0]
        avg_memory_acc = (
            sum(d["memory_transfer_accuracy"] for d in memory_dialogues) / len(memory_dialogues)
            if memory_dialogues else 0.0
        )

        # Policy violations
        total_violations = sum(d["policy_violations"] for d in self.dialogue_results)
        total_turns = sum(d["num_turns"] for d in self.dialogue_results)
        violation_rate = total_violations / total_turns if total_turns > 0 else 0.0

        # Judge scores
        judge_dialogues = [d for d in self.dialogue_results if d.get("avg_judge_score") is not None]
        avg_judge_score = (
            sum(d["avg_judge_score"] for d in judge_dialogues) / len(judge_dialogues)
            if judge_dialogues else None
        )

        return {
            "num_dialogues": num_dialogues,

            # Task completion and system correctness
            "task_success_rate": task_success_rate,
            "avg_system_correctness": avg_system_correctness,

            # Intent and routing
            "avg_intent_accuracy": avg_intent_acc,
            "avg_action_type_accuracy": avg_act_acc,
            "avg_domain_accuracy": avg_domain_acc,

            # Slot tracking
            "avg_jga": avg_jga,
            "avg_slot_accuracy": avg_slot_acc,
            "avg_hallucination_rate": avg_hall_rate,

            # Memory transfer
            "avg_memory_transfer_accuracy": avg_memory_acc,

            # Policy
            "policy_violation_rate": violation_rate,
            "total_policy_violations": total_violations,

            # LLM judge
            "avg_judge_score": avg_judge_score,

            # Detailed dialogue data
            "dialogue_results": self.dialogue_results
        }

