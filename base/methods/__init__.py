from .contrastive import Contrastive
from .byol import BYOL
from .simsiam import SIMSIAM
from .barlow_twins import BARLOW_TWINS
from .vicreg import VICREG
from .bn import BN
from .pca import PCA
from .plain import Plain
from .wmse import WMSE
from .cwrgp import CW_RGP
from .zero_icl import ZeroICL


METHOD_LIST = ["cwrgp","contrastive","zero_icl", "byol", 'simsiam', 'barlowtwins', 'vicreg', 'bn', 'pca', 'plain', 'pca', 'wmse']


def get_method(name):
    assert name in METHOD_LIST
    if name == "contrastive":
        return Contrastive
    elif name == "byol":
        return BYOL
    elif name == "simsiam":
        return SIMSIAM
    elif name == 'barlowtwins':
        return BARLOW_TWINS
    elif name == 'vicreg':
        return VICREG
    elif name == "bn":
        return BN
    elif name == "pca":
        return PCA
    elif name == "plain":
        return Plain
    elif name == "wmse":
        return WMSE
    elif name == "zero_icl":
        return ZeroICL
    elif name == "cwrgp":
        return CW_RGP
    

