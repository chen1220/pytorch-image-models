from .asymmetric_loss import AsymmetricLossMultiLabel, AsymmetricLossSingleLabel
from .binary_cross_entropy import DenseBinaryCrossEntropy
from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .jsd import JsdCrossEntropy
from .focal import Focal_loss
from .eql_loss import SEQLoss