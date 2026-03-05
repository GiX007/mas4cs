"""Tests for all tools (DB query tools)."""
from src.utils import print_separator
from src.core.tools import find_entity, book_entity


def test_match_entity() -> None:
    """Test entity matching logic via a local replica — not imported from tools."""
    print("\n" + "-" * 60)
    print("TEST _match_entity")
    print("-" * 60)

    def match_entity(ent: dict, constraints: dict) -> bool:
        """Local replica of _match_entity for testing purposes."""
        for slot, value in constraints.items():
            if value == "dontcare":
                continue
            field = slot.split("-")[-1]
            if field not in ent:
                continue
            # Simple lowercase normalization for testing (real normalization includes: center→centre, strip whitespace, etc.)
            entity_value = str(ent[field]).lower()
            if entity_value != value:
                return False
        return True

    entity = {
        "name": "acorn guest house",
        "area": "north",
        "pricerange": "moderate",
        "type": "guesthouse",
        "stars": "4",
        "parking": "yes",
        "internet": "yes",
    }
    print(f"\nEntity for testing: {entity}")

    print(f"\nUse of constraints: {{'hotel-area': 'north', 'hotel-pricerange': 'moderate'}} | Result: {match_entity(entity, {'hotel-area': 'north', 'hotel-pricerange': 'moderate'})} | Exact match")
    print(f"Constraints: {{'hotel-area': 'south'}} | Result: {match_entity(entity, {'hotel-area': 'south'})} | Wrong value → no match")
    print(f"Constraints: {{'hotel-pricerange': 'cheap'}} | Result: {match_entity(entity, {'hotel-pricerange': 'cheap'})} | Wrong value → no match")
    print(f"Constraints: {{'hotel-area': 'dontcare', 'hotel-pricerange': 'moderate'}} | Result: {match_entity(entity, {'hotel-area': 'dontcare', 'hotel-pricerange': 'moderate'})} | Matches because area is dontcare")
    print(f"Constraints: {{'hotel-nonexistent': 'value'}} | Result: {match_entity(entity, {'hotel-nonexistent': 'value'})} | Missing slot → matches")
    print(f"Constraints: {{''}} | Result: {match_entity(entity, {})} | No constraints → matches")
    print(f"Constraints: {{'hotel-area': 'dontcare', 'hotel-pricerange': 'dontcare'}} | Result: {match_entity(entity, {'hotel-area': 'dontcare', 'hotel-pricerange': 'dontcare'})} | All dontcare → matches")


def test_find_entity() -> None:
    """Test find_entity public tool."""
    print("\n" + "-" * 60)
    print("TEST find_entity TOOL")
    print("-" * 60)

    # Basic hotel search
    print(f"\nTest 1: Basic Search with constraints {{'hotel-area': 'north', 'hotel-pricerange': 'cheap'}}")
    results = find_entity("hotel", {"hotel-area": "north", "hotel-pricerange": "cheap"})
    print(f"Top 2 results:\n {results[:2]}")

    # Normalization in find_entity
    print(f"\nTest 2: Normalization with constraints {{'hotel-area': 'North', 'hotel-pricerange': 'Cheap'}}")
    results_norm = find_entity("hotel", {"hotel-area": "North", "hotel-pricerange": "Cheap"})
    print(f"Top 2 results:\n {results_norm[:2]}")

    # dontcare
    print(f"\nTest 3: dontcare with constraints {{'hotel-area': 'north', 'hotel-pricerange': 'dontcare'}}")
    results_dc = find_entity("hotel", {"hotel-area": "north", "hotel-pricerange": "dontcare"})
    print(f"Top 2 results:\n {results_dc[:2]}")  # dontcare should return more results than specific pricerange (results)

    # No match
    print(f"\nTest 4: No match with constraints {{'hotel-area': 'north', 'hotel-pricerange': 'expensive', 'hotel-stars': '5'}}")
    results_none = find_entity("hotel", {"hotel-area": "north", "hotel-pricerange": "expensive", "hotel-stars": "5"})
    print(f"Results:\n {results_none}")  # Should be empty list

    # MAX_DB_RESULTS respected
    from src.experiments.config import MAX_DB_RESULTS
    print(f"\nTest 5: MAX_DB_RESULTS respected with no constraints {{}}")
    results_all = find_entity("hotel", {})
    print(f"Number of results returned: {len(results_all)} (should be <= {MAX_DB_RESULTS})")

    # Restaurant search
    print(f"\nTest 6: Restaurant search with constraints {{'restaurant-area': 'centre', 'restaurant-pricerange': 'cheap'}}")
    results_rest = find_entity("restaurant", {"restaurant-area": "centre", "restaurant-pricerange": "cheap"})
    print(f"Top 2 results:\n {results_rest[:2]}")


def test_book_entity() -> None:
    """Test book_entity public tool."""
    print("\n" + "-" * 60)
    print("TEST book_entity TOOL")
    print("-" * 60)

    # Successful booking
    print(f"\nTest 1: Successful booking with constraints {{'hotel-area': 'north', 'hotel-pricerange': 'cheap'}}")
    result = book_entity("hotel", {"hotel-area": "north", "hotel-pricerange": "cheap"})
    print(f"Booking result:\n {result}")

    # Failed booking — no match
    print(f"\nTest 2: Failed booking with constraints {{'hotel-area': 'north', 'hotel-pricerange': 'expensive', 'hotel-stars': '5'}}")
    result_fail = book_entity("hotel", {"hotel-area": "north", "hotel-pricerange": "expensive", "hotel-stars": "5"})
    print(f"Booking result:\n {result_fail}")


def main() -> None:
    """Run all tool tests."""

    print_separator("TEST TOOLS")
    test_match_entity()
    test_find_entity()
    test_book_entity()
    print_separator("END OF TEST TOOLS")


if __name__ == "__main__":
    main()
