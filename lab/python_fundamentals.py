"""
Python fundamentals reference for MAS4CS development.

Hands-on demonstrations of core Python concepts used throughout the project:
- Set operations: intersection, union, difference (used in metric calculations)
- Nested dict navigation (used in slot tracking and JGA)
- Functions vs classes: when to use each (motivates DialogueEvaluator design)
"""

from src.utils import print_separator


def demonstrate_set_operations() -> None:
    """Demonstrate Python set operations for accuracy calculations."""

    print_separator("QUICK REFRESHER IN PYTHON-SETS")

    # Define some Lists to play with
    predicted_single = ["Restaurant-Book"]
    ground_truth_single = ["Hotel-Book"]
    predicted_mult = ["Hotel-Inform", "Hotel-Request"]
    ground_truth_mult = ["Hotel-Request", "Restaurant-Inform"]
    print("\nLists:")
    print(f"predicted_single: {predicted_single}")
    print(f"ground_truth_single: {ground_truth_single}")
    print(f"predicted_mult: {predicted_mult}")
    print(f"ground_truth_mult: {ground_truth_mult}")

    # Example 1: Converting to set
    print("\n1. CONVERTING TO SETS")
    predicted_single_set = set(predicted_single)
    ground_truth_single_set = set(ground_truth_single)
    predicted_mult_set = set(predicted_mult)
    ground_truth_mult_set = set(ground_truth_mult)
    print(f"predicted_single set: {predicted_single_set}")
    print(f"ground_truth_single set: {ground_truth_single_set}")
    print(f"predicted_mult set: {predicted_mult_set}")
    print(f"ground_truth_mult set: {ground_truth_mult_set}")

    # Example 2: Sets remove duplicates automatically
    print("\n2. SETS REMOVE DUPLICATES AUTOMATICALLY")
    duplicates = ["Hotel-Inform", "Hotel-Inform", "Hotel-Request"]
    unique_set = set(duplicates)
    print(f"List with duplicates: {duplicates}")
    print(f"Set (unique only): {unique_set}")

    # Example 3: Sets ignore order
    print("\n3. SETS IGNORE ORDER")
    list_a = ["Hotel-Inform", "Hotel-Request"]
    list_b = ["Hotel-Request", "Hotel-Inform"]  # Different order
    set_a = set(list_a)
    set_b = set(list_b)
    print(f"List A: {list_a} | Set A: {set_a}")
    print(f"List B: {list_b} | Set B: {set_b}")
    print(f"Sets equal? {set_a == set_b}")

    # Example 4: Intersection (& operator) - common elements
    print("\n4. INTERSECTION (&) - WHAT'S COMMON?")
    predicted_set = {"Hotel-Inform", "Hotel-Request"}
    ground_truth_set = {"Hotel-Request", "Restaurant-Inform"}

    intersection = predicted_set & ground_truth_set
    print(f"Predicted: {predicted_set}")
    print(f"Ground truth: {ground_truth_set}")
    print(f"Intersection: {intersection}")  # {"Hotel-Request"}
    print(f"Common items: {len(intersection)}")  # 1

    # Example 5: Union (| operator) - all unique elements
    print("\n5. UNION (|) - ALL UNIQUE ITEMS")
    union = predicted_set | ground_truth_set
    print(f"Predicted: {predicted_set}")
    print(f"Ground truth: {ground_truth_set}")
    print(f"Union: {union}")
    print(f"Total unique: {len(union)}")

    # Example 6: Difference (-) - what's in A, but not B
    print("\n6. DIFFERENCE (-) - WHAT'S MISSING(A - B)?")
    only_in_predicted = predicted_set - ground_truth_set
    only_in_ground_truth = ground_truth_set - predicted_set
    print(f"Predicted: {predicted_set}")
    print(f"Ground truth: {ground_truth_set}")
    print(f"Only in predicted: {only_in_predicted}")  # {"Hotel-Inform"}
    print(f"Only in ground truth: {only_in_ground_truth}")  # {"Restaurant-Inform"}

    # Example 7: Practical accuracy calculation
    print("\n7. PRACTICAL ACCURACY CALCULATION")
    predicted = ["Restaurant-Inform"]
    ground_truth = ["Restaurant-Inform", "Restaurant-Request"]

    predicted_set = set(predicted)
    ground_truth_set = set(ground_truth)

    # Exact match (strict)
    exact_match = predicted_set == ground_truth_set
    print(f"Predicted: {predicted_set}")
    print(f"Ground truth: {ground_truth_set}")
    print(f"Exact match? {exact_match}")  # False

    # Recall and Precision
    correct = predicted_set & ground_truth_set
    num_correct = len(correct)
    num_predicted = len(predicted_set)
    num_ground_truth = len(ground_truth_set)

    recall = num_correct / num_ground_truth if num_ground_truth > 0 else 0.0
    precision = num_correct / num_predicted if num_predicted > 0 else 0.0

    print(f"Correct items: {correct}")
    print(f"Recall: {recall:.2f} ({num_correct}/{num_ground_truth})")
    print(f"Precision: {precision:.2f} ({num_correct}/{num_predicted})")

    # Example 8: Empty sets
    print("\n8. EDGE CASE - EMPTY SETS")
    empty_set = set([])
    non_empty = {"Hotel-Inform"}
    print(f"Empty set: {empty_set}")
    print(f"Non-empty: {non_empty}")
    print(f"Intersection: {empty_set & non_empty}")  # set()
    print(f"Are they equal? {empty_set == non_empty}")  # False

    print_separator("END OF QUICK REFRESHER IN PYTHON-SETS")


def demonstrate_nested_dict_navigation() -> None:
    """Demonstrate how to navigate nested dictionaries for slot comparisons."""

    print_separator("NESTED DICTIONARY NAVIGATION - SLOT TRACKING")

    # Example slot dictionaries (same structure as MultiWOZ)
    predicted_slots = {
        "hotel": {"area": "centre", "pricerange": "cheap"},
        "restaurant": {"food": "italian"}
    }

    ground_truth_slots = {
        "hotel": {"area": "centre", "pricerange": "expensive"},
        "restaurant": {"food": "italian", "area": "north"}
    }

    print(f"\nPredicted slots:  {predicted_slots}")
    print(f"Ground truth slots: {ground_truth_slots}")

    print("\n" + "-" * 60)
    print("Step-by-step Navigation (Slot Accuracy Logic)")
    print("-" * 60)
    print("\nInspect: ground_truth_slots dictionary\n")
    print(f"Dict keys: {ground_truth_slots.keys()}")
    print("\nLoop over keys (outer dict):")
    num_correct = 0
    num_total = 0

    # Outer loop: iterate through ground_truth_slots dict
    for domain, slots in ground_truth_slots.items():
        print(f"\nKey (inner dict): {domain}")
        print(f"Value: {slots}")
        print(f"Type of value: {type(slots)}")

        # Inner loop: iterate through each slot-value pair in this domain
        for slot, value in slots.items():
            num_total += 1
            print(f"\n    Key: {slot}")
            print(f"    Value: {value} | Type: {type(value)}\n")

            # Check if predicted has this domain
            if domain in predicted_slots:
                print(f"    Key(domain) '{domain}' exists in predictions")
                print(f"    Predicted slots for {domain}: {predicted_slots[domain]}")

                # Check if predicted has this exact slot-value pair
                predicted_value = predicted_slots[domain].get(slot)
                print(f"    Predicted value for '{slot}': {predicted_value}")

                if predicted_value == value:
                    num_correct += 1
                    print(f"      MATCH! '{slot}': {predicted_value} == {value}")
                    print(f"      Correct slots so far: {num_correct}")
                else:
                    print(f"      MISMATCH! '{slot}': {predicted_value} != {value}")
            else:
                print(f"    Key(domain) '{domain}' NOT in predictions")
                print(f"    This slot-value pair is missing entirely")

    # Calculate accuracy
    accuracy = num_correct / num_total if num_total > 0 else 0.0

    print("\n" + "-" * 60)
    print("Summary")
    print("-" * 60)
    print(f"Correct slot-value pairs: {num_correct}")
    print(f"Total ground truth pairs: {num_total}")
    print(f"Slot Accuracy: {accuracy:.2f} ({num_correct}/{num_total})")

    print_separator("END OF NESTED DICTIONARY NAVIGATION TESTS")


def demonstrate_class_vs_function() -> None:
    """
    Demonstrate when to use classes vs functions.
    Shows why DialogueEvaluator is a class, not a function.
    """

    print_separator("CLASSES VS FUNCTIONS")

    # Example 1: FUNCTION (Stateless)
    print("\n" + "-" * 60)
    print("Example 1: FUNCTION (Stateless Calculation)")
    print("-" * 60)

    def calculate_average(numbers):
        """Pure function: input → output, no memory."""
        return sum(numbers) / len(numbers) if numbers else 0

    print("\nCalculating averages with a function:")
    print(f"  Average of [1, 2, 3]: {calculate_average([1, 2, 3])}")
    print(f"  Average of [10, 20]: {calculate_average([10, 20])}")
    print("\nFUNCTION is perfect here - no state needed!")


    # Example 2: CLASS (Stateful Accumulator)
    print("\n" + "-" * 60)
    print("Example 2: CLASS (Stateful Accumulator)")
    print("-" * 60)

    class RunningAverage:
        """Class: remembers all numbers added so far."""

        def __init__(self):
            self.numbers = []  # State: stored inside object

        def add(self, number):
            """Add a number to the running collection."""
            self.numbers.append(number)

        def get_average(self):
            """Calculate average of all added numbers."""
            return sum(self.numbers) / len(self.numbers) if self.numbers else 0

        def reset(self):
            """Clear all numbers."""
            self.numbers = []

    print("\nCalculating running average with a class:")
    avg = RunningAverage()

    avg.add(1)
    print(f"  Added 1, current average: {avg.get_average()}")

    avg.add(2)
    print(f"  Added 2, current average: {avg.get_average()}")

    avg.add(3)
    print(f"  Added 3, current average: {avg.get_average()}")

    print(f"\n  All numbers stored: {avg.numbers}")
    print("\nCLASS is perfect here - state accumulated automatically!")


    # Example 3: Function approach of DialogueEvaluator  (manual state)
    print("\n" + "-" * 60)
    print("Example 3: Function Approach of DialogueEvaluator (Manual State Tracking)")
    print("-" * 60)

    def evaluate_turn_function(predicted, truth):
        """Evaluate one turn - returns score but doesn't remember anything."""
        return 1.0 if predicted == truth else 0.0

    def evaluate_dialogue_function(scores):
        """Calculate average from a list of scores."""
        if not turn_scores:
            return 0.0
        return sum(scores) / len(scores)

    print("\nEvaluating a dialogue with multiple turns (FUNCTION approach):")

    # List for manually tracking state
    turn_scores = []

    print("\n  Turn 1:")
    score1 = evaluate_turn_function(predicted="find_hotel", truth="find_hotel")
    turn_scores.append(score1)
    print(f"    Predicted: find_hotel, Truth: find_hotel → Score: {score1}")

    print("\n  Turn 2:")
    score2 = evaluate_turn_function(predicted="book_hotel", truth="find_restaurant")
    turn_scores.append(score2)
    print(f"    Predicted: book_hotel, Truth: find_restaurant → Score: {score2}")

    print("\n  Turn 3:")
    score3 = evaluate_turn_function(predicted="book_hotel", truth="book_hotel")
    turn_scores.append(score3)
    print(f"    Predicted: book_hotel, Truth: book_hotel → Score: {score3}")

    dialogue_score = evaluate_dialogue_function(turn_scores)
    print(f"\n  Dialogue average: {dialogue_score:.2f} ({sum(turn_scores)}/{len(turn_scores)} correct)")

    print("\nProblems with function approach:")
    print("   - You must manually track 'turn_scores' list")
    print("   - Easy to forget to append a score")
    print("   - Have to pass 'turn_scores' to evaluate_dialogue_function")
    print("   - State is scattered (score1, score2, score3, turn_scores all separate)")


    # Example 4: Why DialogueEvaluator is a Class
    print("\n" + "-" * 60)
    print("Example 4: Class Approach of DialogueEvaluator (Automatic State Tracking)")
    print("-" * 60)

    class SimpleDialogueEvaluator:
        """Simplified version of DialogueEvaluator to show the pattern."""

        def __init__(self):
            self.turn_scores = []  # Accumulates scores across turns

        def evaluate_turn(self, predicted, truth):
            """Evaluate one turn and remember the score."""
            score = 1.0 if predicted == truth else 0.0
            self.turn_scores.append(score)  # Save the score to state of the class
            return score

        def evaluate_dialogue(self):
            """Get average score across all evaluated turns."""
            if not self.turn_scores:
                return 0.0
            return sum(self.turn_scores) / len(self.turn_scores)

        def reset(self):
            """Clear scores for next dialogue."""
            self.turn_scores = []

    print("\nEvaluating a dialogue with multiple turns (CLASS approach):")
    evaluator = SimpleDialogueEvaluator()
    # At this moment: evaluator.turn_scores = []

    print("\n  Turn 1:")
    score1 = evaluator.evaluate_turn(predicted="find_hotel", truth="find_hotel")
    print(f"    Predicted: find_hotel, Truth: find_hotel → Score: {score1}")
    # Inside the method: score = 1.0
    # Then: self.turn_scores.append(1.0)
    # Now: evaluator.turn_scores = [1.0]
    # Returns: 1.0 (stored in score1 variable)

    print("\n  Turn 2:")
    score2 = evaluator.evaluate_turn(predicted="book_hotel", truth="find_restaurant")
    print(f"    Predicted: book_hotel, Truth: find_restaurant → Score: {score2}")
    # Inside the method: score = 0.0
    # Then: self.turn_scores.append(0.0)
    # Now: evaluator.turn_scores = [1.0, 0.0]
    # Returns: 0.0 (stored in score2 variable)

    print("\n  Turn 3:")
    score3 = evaluator.evaluate_turn(predicted="book_hotel", truth="book_hotel")
    print(f"    Predicted: book_hotel, Truth: book_hotel → Score: {score3}")
    # Inside the method: score = 1.0
    # Then: self.turn_scores.append(1.0)
    # Now: evaluator.turn_scores = [1.0, 0.0, 1.0]
    # Returns: 1.0 (stored in score3 variable)

    # The score1, score2, score3 variables are just copies of the return values
    # The REAL data is stored in self.turn_scores inside the object

    dialogue_score = evaluator.evaluate_dialogue()
    print(f"\n  Dialogue average: {dialogue_score:.2f} ({sum(evaluator.turn_scores)}/{len(evaluator.turn_scores)} correct)")

    print("\nBenefits of class approach:")
    print("   - Class automatically tracks state (self.turn_scores)")
    print("   - No manual append needed - happens inside evaluate_turn()")
    print("   - evaluate_dialogue() already knows the scores")
    print("   - State is encapsulated (hidden inside the object)")
    print("   - Can reset() for next dialogue cleanly")
    print("   - Clean, safe, state hidden inside object")
    print("   - Use Class approach when State is complex (not just a list - like in real DialogueEvaluator with turn_metrics, dialogue_history, etc.)")


    print_separator("END OF CLASSES VS FUNCTIONS")

    # Use FUNCTION if: No state to remember (pure calculation)
    # Examples: calculate_jga(), calculate_intent_accuracy()
    # Use CLASS if: Need to remember things between calls
    # Examples: DialogueEvaluator, DatasetEvaluator


def run_all_demonstrations() -> None:
    """Run all utility demonstrations."""
    demonstrate_set_operations()
    demonstrate_nested_dict_navigation()
    demonstrate_class_vs_function()


if __name__ == "__main__":
    run_all_demonstrations()


