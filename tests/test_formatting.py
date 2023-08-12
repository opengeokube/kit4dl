from unittest.mock import patch

import pytest

try:
    import tomllib as toml
except ModuleNotFoundError:
    import toml  # type: ignore[no-redef]

from kit4dl.formatting import (
    _UNIX_PATHNAME_SEP,
    _WINDOWS_PATHNAME_SEP,
    escape_os_sep,
    substitute_symbols,
)


class TestFormatting:
    @pytest.fixture
    def load_path(self) -> str:
        import os
        from uuid import uuid4

        load = """
        [section]
        target = "{{ PROJECT_DIR }}/data.nc"
        """
        path = f"{uuid4()}.toml"
        with open(path, "wt") as file:
            file.writelines(load)
        yield path
        os.remove(path)

    @pytest.fixture
    def load_path_env(self) -> str:
        import os
        from uuid import uuid4

        load = """
        [section]
        target = "{{ env['env_path'] }}/data.nc"
        """
        path = f"{uuid4()}.toml"
        with open(path, "wt") as file:
            file.writelines(load)
        yield path
        os.remove(path)

    def test_substitute_symbols(self, load_path):
        result = substitute_symbols(load_path, PROJECT_DIR="/work/usr")
        entries = toml.loads(result)
        assert entries["section"]["target"] == "/work/usr/data.nc"

    def test_fail_on_missing_placeholder_definition(self, load_path):
        with pytest.raises(ValueError, match=r"'PROJECT_DIR' is undefined"):
            _ = substitute_symbols(load_path)

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

    def test_use_env_var_in_conf(self, load_path_env):
        import os

        os.environ["env_path"] = "/usr/new_path"
        result = substitute_symbols(load_path_env)
        entries = toml.loads(result)
        assert entries["section"]["target"] == "/usr/new_path/data.nc"
