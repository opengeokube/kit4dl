import os
import pytest
from unittest.mock import patch


from kit4dl.mixins import ObfuscateKeyMixing, LoggerMixin
from pydantic import BaseModel, Field


class TestLoggingMixin:
    @pytest.mark.parametrize(
        "log_method", ["debug", "info", "warn", "error", "critical"]
    )
    def test_log_methods_are_available(self, log_method):
        class A(LoggerMixin):
            pass

        assert hasattr(A(), log_method)

    def test_obfuscate_mixin_inner_dict(self):
        def new_dict():
            return {
                "my_key": "aaa",
                "key_2": "ccc",
                "key": "ddd",
                "aaa": "mmm",
            }

        class A(BaseModel, ObfuscateKeyMixing):
            c: dict = Field(default_factory=new_dict)

        a = A()
        dump = a.obfuscated_dict()
        assert (
            dump["c"]["my_key"] == ObfuscateKeyMixing.KIT4DL_OBFUSCATING_VALUE
        )
        assert (
            dump["c"]["key_2"] == ObfuscateKeyMixing.KIT4DL_OBFUSCATING_VALUE
        )
        assert dump["c"]["key"] == ObfuscateKeyMixing.KIT4DL_OBFUSCATING_VALUE
        assert dump["c"]["aaa"] == "mmm"

    def test_obfuscate_key(self):
        class A(BaseModel, ObfuscateKeyMixing):
            a: str = "a"
            b: tuple = (10, 20)
            a_key: str = "1234"
            key_2: str = "1234"

        a = A()
        dump = a.obfuscated_dict()
        assert dump["a"] == "a"
        assert dump["b"] == (10, 20)
        assert dump["a_key"] == ObfuscateKeyMixing.KIT4DL_OBFUSCATING_VALUE
        assert dump["key_2"] == ObfuscateKeyMixing.KIT4DL_OBFUSCATING_VALUE

    @patch.dict(os.environ, {"KIT4DL_KEY_TO_OBFUSCATE": "mine"})
    def test_obfuscate_key_from_venv(self):
        class A(BaseModel, ObfuscateKeyMixing):
            mine: str = "a"
            key: tuple = (10, 20)

        a = A()
        dump = a.obfuscated_dict()
        assert dump["mine"] == ObfuscateKeyMixing.KIT4DL_OBFUSCATING_VALUE
        assert dump["key"] == (10, 20)

    @patch.dict(os.environ, {"KIT4DL_OBFUSCATING_VALUE": "aaa"})
    def test_obfuscate_key_from_venv(self):
        class A(BaseModel, ObfuscateKeyMixing):
            mine: str = "a"
            key: tuple = (10, 20)

        a = A()
        dump = a.obfuscated_dict()
        assert dump["mine"] == "a"
        assert dump["key"] == "aaa"

    def test_fail_bfuscate_mixin_on_nonpydantic_basemodel_subclass(self):
        class A(ObfuscateKeyMixing):
            mine: str = "a"
            key: tuple = (10, 20)
