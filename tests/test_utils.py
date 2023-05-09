import pytest
import torch

skipnocuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() == 0,
    reason="no cuda or no GPU available",
)
