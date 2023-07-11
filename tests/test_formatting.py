from unittest.mock import patch

import pytest

try:
    import tomllib as toml
except ModuleNotFoundError:
    import toml  # type: ignore[no-redef]

from mlkit.formatting import (
    _UNIX_PATHNAME_SEP,
    _WINDOWS_PATHNAME_SEP,
    escape_os_sep,
    substitute_symbols,
)


class TestFormatting:
    @pytest.fixture
    def load(self) -> str:
        return """
        [section]
        target = "{{ PROJECT_DIR }}/data.nc"
        """

    def test_substitute_symbols(self, load):
        result = substitute_symbols(load, PROJECT_DIR="/work/usr")
        entries = toml.loads(result)
        assert entries["section"]["target"] == "/work/usr/data.nc"

    def test_fail_on_missing_placeholder_definition(self, load):
        with pytest.raises(ValueError, match=r"'PROJECT_DIR' is undefined"):
            _ = substitute_symbols(load)

    @patch("os.sep", _UNIX_PATHNAME_SEP)
    def test_substitute_windows_path_to_unix(self):
        load = _WINDOWS_PATHNAME_SEP.join(
            ["D:", "Program Files", "some_file.txt"]
        )
        escaped = escape_os_sep(load)
        assert _WINDOWS_PATHNAME_SEP not in escaped
        assert _UNIX_PATHNAME_SEP in escaped

    @patch("os.sep", _WINDOWS_PATHNAME_SEP)
    def test_substitute_unix_path_to_windows(self):
        load = _UNIX_PATHNAME_SEP.join(
            ["D:", "Program Files", "some_file.txt"]
        )
        escaped = escape_os_sep(load)
        assert _UNIX_PATHNAME_SEP not in escaped
        assert _WINDOWS_PATHNAME_SEP in escaped

    @patch("os.sep", _WINDOWS_PATHNAME_SEP)
    def test_making_uniform_os_sep_windows(self):
        load = "D:\\Program Files/dir1/dir2\\file.txt"
        escaped = escape_os_sep(load)
        assert escaped == "D:\\\\Program Files\\\\dir1\\\\dir2\\\\file.txt"

    @patch("os.sep", _UNIX_PATHNAME_SEP)
    def test_making_uniform_os_sep_unix(self):
        load = "D:\\Program Files/dir1/dir2\\file.txt"
        escaped = escape_os_sep(load)
        assert escaped == "D:/Program Files/dir1/dir2/file.txt"

    def test_use_env_var_in_conf(self):
        import os

        os.environ["env_path"] = "/usr/new_path"
        load: str = """
        [section]
        target = "{{ env['env_path'] }}/data.nc"
        """
        result = substitute_symbols(load)
        entries = toml.loads(result)
        assert entries["section"]["target"] == "/usr/new_path/data.nc"
