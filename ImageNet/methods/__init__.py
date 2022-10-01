from .wmse import WMSE
from .cwrgp import CW_RGP

METHOD_LIST = ["cwrgp", "wmse"]


def get_method(name):
    assert name in METHOD_LIST
    if name == "wmse":
        return WMSE
    elif name == "cwrgp":
        return CW_RGP

