try:
    import pytorch_lightning
except ImportError:
    msg = """
This package requires pytorch-lightning. You can install the extra dependencies with:

    pip install 'finite-element-networks[lightning]'
""".strip()
    raise RuntimeError(msg)

from .callbacks import MultipleShootingCallback
from .data import (
    BlackSeaDataModule,
    CylinderFlowDataModule,
    MeshConfig,
    ScalarFlowDataModule,
)
from .regression import SequenceRegressionTask
