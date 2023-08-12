import pytest
import torch

from kit4dl.nn import validators
from tests.utils import skipnocuda


class TestValidators:
    def test_validate_cuda_device_exists_none(self):
        assert validators.validate_cuda_device_exists(None) is None

    @skipnocuda
    def test_validate_cuda_device_id_too_big(self):
        cuda_id = torch.cuda.device_count() + 1
        with pytest.raises(
            AssertionError,
            match=f"CUDA device with id `{cuda_id}` does not exist",
        ):
            validators.validate_cuda_device_exists(cuda_id)

    def test_validate_class_exists_fail_on_not_exising_qualified_path(self):
        with pytest.raises(ModuleNotFoundError, match=r"No module named 'a'"):
            validators.validate_class_exists("a.b::c")
