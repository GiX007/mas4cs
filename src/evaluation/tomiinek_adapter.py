"""
Adapter to convert MAS4CS dialogue results into the Tomiinek evaluator input format.

(https://github.com/Tomiinek/MultiWOZ_Evaluation)

Metrics computed:
- Inform%: % of dialogues where the system provided an entity matching all user constraints
- Success%: % of dialogues where the system also answered all requested attributes (e.g. phone, address)
- BLEU: n-gram overlap between system responses and MultiWOZ 2.2 reference responses (mwz22 style)
- Combined: Standard MultiWOZ summary score = 0.5 * (Inform + Success) + BLEU
"""
from mwzeval.metrics import Evaluator


def build_tomiinek_input(dialogue_results: list[dict]) -> dict[str, list[dict]]:
    """
    Convert MAS4CS dialogue results to Tomiinek evaluator input format.

    Each dialogue produces one entry per turn, containing the system response and the predicted belief state at that turn.

    Args:
        dialogue_results: List of dialogue result dicts from DialogueEvaluator. Each must contain 'dialogue_id' and 'turn_metrics'.

    Returns:
        Dict mapping normalized dialogue id -> list of per-turn dicts, ready to pass to Tomiinek evaluator.
    """
    predictions = {}

    for dialogue in dialogue_results:
        # Normalize: 'MUL1271.json' -> 'mul1271'
        raw_id = dialogue.get("dialogue_id", "")
        dialogue_id = raw_id.lower().replace(".json", "")

        turns = []
        for turn in dialogue.get("turn_metrics", []):
            # Belief state: Tomiinek expects state as {domain: {slot: value}} -> our accumulated slots are already in this shape
            # predicted_slots = turn.get("predicted_slots", {})

            turns.append({
                "response": turn.get("system_response", ""),
                # "state": predicted_slots,  # Omitted: Tomiinek automatically uses GT belief state from MultiWOZ 2.2
                "active_domains": [turn.get("domain")] if turn.get("domain") else [],
            })

        if turns:
            predictions[dialogue_id] = turns

    return predictions

def compute_tomiinek_metrics(dialogue_results: list[dict]) -> dict[str, float]:
    """
    Convert dialogue results to Tomiinek format and compute official MultiWOZ metrics.

    Calls the Tomiinek evaluator once on the full prediction set and extracts Inform, Success, BLEU, and Combined scores.

    Args:
        dialogue_results: List of dialogue result dicts from DialogueEvaluator.

    Returns:
        Dict with keys: 'inform_rate', 'success_rate', 'bleu', 'combined'.
    """
    # Bring dialogue results into Tomiinek input format
    predictions = build_tomiinek_input(dialogue_results)
    # print(f"\nPredictions: {predictions}")

    if not predictions:
        return {"inform_rate": 0.0, "success_rate": 0.0, "bleu": 0.0, "combined": 0.0}

    try:
        e = Evaluator(bleu=True, success=True, richness=False)
        results = e.evaluate(predictions)
        # print(f"\nTomiinek raw results: {results}")

        inform = results["success"]["inform"]["total"]
        success = results["success"]["success"]["total"]
        bleu = results["bleu"]["mwz22"]
        combined = 0.5 * (inform + success) + bleu

        return {
            "inform_rate": round(inform, 4),
            "success_rate": round(success, 4),
            "bleu": round(bleu, 4),
            "combined": round(combined, 4),
        }

    except Exception as e:
        print(f"Tomiinek evaluation failed: {e}")
        return {"inform_rate": 0.0, "success_rate": 0.0, "bleu": 0.0, "combined": 0.0}
