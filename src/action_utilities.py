from action_records import BasicAction

def is_insert(action: BasicAction) -> bool:
    return action.get_name() == "insert"

def create_insert_action(text: str) -> BasicAction:
    return BasicAction("insert", [text])

def get_insert_text(action: BasicAction) -> str:
    return action.get_arguments()[0]

def is_insert_only_actions(actions: list[BasicAction]):
    return len(actions) == 1 and is_insert(actions[0])

def get_insert_text_from_insert_only_actions(actions: list[BasicAction]):
    return get_insert_text(actions[0])