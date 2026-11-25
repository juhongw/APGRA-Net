from __future__ import annotations

from .cylfbnet import CylFBNet


class APGRANet(CylFBNet):
    """Deployment network used by the APGRA training framework.

    The compact periodic encoder--decoder is retained as the segmentation
    backbone. APGRA adds geometry-decoupled proxy initialization and the
    target-domain adaptation objectives implemented in train_adaptation.py.
    """

