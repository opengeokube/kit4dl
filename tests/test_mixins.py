import pytest


class TestLoggingMixin:
    @pytest.mark.parametrize(
        "log_method", ["debug", "info", "warn", "error", "critical"]
    )
    def test_log_methods_are_available(self, log_method):
        from kit4dl.mixins import LoggerMixin

        class A(LoggerMixin):
            pass

        assert hasattr(A(), log_method)
