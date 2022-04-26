def action_type_to_number(type: str) -> int:
    if type == "mousedown":
        return 0
    elif type == "mouseup":
        return 1
    elif type == "click":
        return 2
    else:
        raise NotImplemented


def action_number_to_type(action: int) -> str:
    if action == 0:
        return "mousedown"
    elif action == 1:
        return "mouseup"
    elif action == 2:
        return "click"
    else:
        raise NotImplemented
