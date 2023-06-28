import pytest
import toml

from mlkit.formatting import substitute_symbols

class TestFormatting:

    @pytest.fixture
    def load(self) -> str:
        return """
        [section]
        target = "{PROJECT_DIR}/data.nc"
        """

    def test_substitute_symbols(self, load):
        result = substitute_symbols(load, PROJECT_DIR="/work/usr")
        entries = toml.loads(result)
        assert entries["section"]["target"] == "/work/usr/data.nc"

    def test_fail_on_missing_placeholder_definition(self, load):
        with pytest.raises(KeyError, match=r"no value found for the placeholder `PROJECT_DIR`"):
            _ = substitute_symbols(load, NOT_EXISTING="/work/usr")            
