"""Field validation shared by service trace capture and reduction."""


def _object(value, field):
    if not isinstance(value, dict):
        raise ValueError(f"{field} 必须是对象")
    return value


def _list(value, field):
    if not isinstance(value, list):
        raise ValueError(f"{field} 必须是数组")
    return value


def _text(obj, key, field):
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}.{key} 必须是非空字符串")
    return value


def _integer(obj, key, field):
    value = obj.get(key)
    if type(value) is not int or not 0 <= value < 2**63:
        raise ValueError(f"{field}.{key} 必须是非负 64 位整数")
    return value


def _tokens(value, field):
    tokens = _list(value, field)
    if any(type(token) is not int or not 0 <= token < 2**31 for token in tokens):
        raise ValueError(f"{field} 含无效 token ID")
    return tokens


def _check_round_counts(proposed, verified, accepted, limit, field):
    if any(type(n) is not int for n in (proposed, verified, accepted)) or not (
        0 <= accepted <= verified <= proposed <= limit
    ):
        raise ValueError(f"{field} 须满足 accepted ≤ verified ≤ proposed ≤ configured_k")
