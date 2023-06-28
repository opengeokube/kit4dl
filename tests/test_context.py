import pytest

from mlkit import context


class TestSession:
    @pytest.fixture(autouse=True, scope="session")
    def set_session_attributes(self):
        context.LOG_LEVEL = "info"
        context.LOG_FORMAT = "..."
        context.PROJECT_DIR = "/work/my_dir"

    @pytest.mark.parametrize(
        "session_attr", ["LOG_LEVEL", "LOG_FORMAT", "PROJECT_DIR"]
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