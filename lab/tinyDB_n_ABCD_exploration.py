# Tests for validating core functionality and behavior across multiple modules.
from tinydb import TinyDB, Query
import json

def tinydb_test():
    db_path = "C:/Users/giorg/Projects/PycharmProjects/mas4cs/lab_data/test_db.json"

    # Create / open DB
    db = TinyDB(db_path)

    # Reset / clear DB
    db.truncate()

    User = Query()

    # Insert single
    db.insert({'name': 'John', 'age': 22})
    db.insert({'name': 'Alice', 'age': 32})

    # Insert multiple
    db.insert_multiple([
        {'name': 'Theo', 'age': 40},
        {'name': 'Steve', 'age': 38}
    ])

    # Search
    johns = db.search(User.name == 'John')
    marys = db.search(User.name == 'Mary')
    print("Is there any John in the dataset? -> ", johns)
    print("Is there any Mary in the dataset? -> ", marys)

    # Update
    db.update({'age': 23}, User.name == 'John')

    # Remove
    db.remove(User.name == 'Alice')

    # View everything
    all_rows = db.all()

    # Count
    count = len(db)

    # Output (smoke-test style)
    print('All rows:', all_rows)
    print('Row count:', count)


def test_abcd():
    # https://arxiv.org/pdf/2104.00783
    # https://github.com/asappresearch/abcd
    data_path = "C:/Users/giorg/Projects/PycharmProjects/mas4cs/lab_data/abcd_sample.json"

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"ABCD sample type:{type(data)}, length: {len(data)}")
    print("Each record/element is a conversation."); print()

    # print(data[0])
    first = data[0]
    print("First element's type:", type(first))
    print("First element's keys:", first.keys()); print() # ['convo_id', 'scenario', 'original', 'delexed']

    # Unpack each dict
    convo_id = first["convo_id"]
    scenario = first["scenario"]
    original_dialogue = first["original"]
    delexed_dialogue = first["delexed"]

    print(type(convo_id)); print(type(scenario)); print(type(original_dialogue)); print(type(delexed_dialogue)); print()

    # Unpack scenario
    print("Unpack scenario:")
    print(scenario.keys())
    for k, v in scenario.items():
        print(k, v)

    # Unpack original_dialogue
    print("\nUnpack original dialogue:")
    print("List length:", len(original_dialogue)); print()
    first_orig_dial = original_dialogue[0]
    speaker, text = first_orig_dial
    second_orig_dial = original_dialogue[1]
    third_orig_dial = original_dialogue[2]
    print("1st element:", first_orig_dial); print(); print(f"Speaker: {speaker}, Text: {text}"); print(); print("2nd element:", second_orig_dial); print(); print("3rd element:", third_orig_dial); print()

    # Unpack delexed dialogue
    print("Unpack dialogue:")
    print("List length:", len(delexed_dialogue)); print()
    first_del_dial = delexed_dialogue[0]
    second_del_dial = delexed_dialogue[1]
    third_del_dial = delexed_dialogue[2]
    print("1st element:", first_del_dial); print(); print(f"Speaker: {first_del_dial['speaker']}, Text: {first_del_dial['text']}, Turn Count: {first_del_dial['turn_count']}, Targets: {first_del_dial['targets']}, Candidates: {first_del_dial['candidates']}"); print(); print("2nd element:", second_del_dial); print(); print("3rd element:", third_del_dial); print()


if __name__ == "__main__":

    # Test a tiny db
    tinydb_test()

    # Explore ABCD dataset for customer service (products)
    test_abcd()


