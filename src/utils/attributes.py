from functools import reduce
from typing import Any


def get_attribute(
    obj: object, attribute: str, default: Any = None, raise_exception: bool = False
) -> Any:
    """Accept a dotted path to a nested attribute to get."""

    try:
        return reduce(getattr, attribute.split("."), obj)
    except AttributeError as e:
        if raise_exception:
            raise e
        return default


def set_attribute(obj: object, attribute: str, value: Any) -> Any:
    """Accept a dotted path to a nested attribute to set."""

    path, _, target = attribute.rpartition(".")
    if path:
        for attr_name in path.split("."):
            obj = getattr(obj, attr_name)
    setattr(obj, target, value)