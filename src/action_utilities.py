from action_records import BasicAction

def is_insert(action: BasicAction) -> bool:
    return action.get_name() == "insert"

def create_insert_action(text: str) -> BasicAction:
    return BasicAction("insert", [text])