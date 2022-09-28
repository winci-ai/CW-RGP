from .wmse import WMSE
from .cwrg import CWRG

METHOD_LIST = ["cwrg", "wmse"]


def get_method(name):
    assert name in METHOD_LIST
    if name == "wmse":
        return WMSE
    elif name == "cwrg":
        return CWRG

