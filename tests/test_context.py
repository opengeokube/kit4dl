from unittest.mock import patch

import pytest

from kit4dl import context


class TestSession:
    @pytest.fixture(autouse=True, scope="class")
    def set_session_attributes(self):
        context.LOG_LEVEL = "INFO"
        context.LOG_FORMAT = "%(asctime)s"
        context.PROJECT_DIR = "/work/my_dir"
        context.VERSION = "0.0.1"
        yield

    @pytest.mark.parametrize(
        "session_attr", ["LOG_LEVEL", "LOG_FORMAT", "PROJECT_DIR", "VERSION"]
    )
    def test_fail_on_successive_session_attribute_set(self, session_attr):
        with pytest.raises(
            RuntimeError, match=r"Session properties can be set only once"
        ):
            setattr(context, session_attr, "new_value")

    def test_fail_on_creating_new_session_attribute(self):
        with pytest.raises(
            RuntimeError, match=r"Cannot set new session property!"
        ):
            context.NEW_ATTR = 10

    def test_attribute_availability_when_setting_by_other_thread(self):
        # TODO
        pass
