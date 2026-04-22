"""Tests for the bundled OpenAI image_gen plugin."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Import the plugin module directly (not via the plugin loader) so we can
# unit-test the provider class without spinning up PluginManager.
import plugins.image_gen.openai as openai_plugin


@pytest.fixture(autouse=True)
def _tmp_hermes_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return openai_plugin.OpenAIImageGenProvider()


def _fake_response(*, b64: str | None = None, url: str | None = None,
                   revised_prompt: str | None = None):
    """Build a stand-in for ``client.images.generate()`` return value."""
    item = SimpleNamespace(b64_json=b64, url=url, revised_prompt=revised_prompt)
    return SimpleNamespace(data=[item])


class TestMetadata:
    def test_name(self, provider):
        assert provider.name == "openai"

    def test_display_name(self, provider):
        assert provider.display_name == "OpenAI"

    def test_list_models_contains_defaults(self, provider):
        ids = {m["id"] for m in provider.list_models()}
        assert "gpt-image-1.5" in ids
        assert "dall-e-3" in ids
        assert "dall-e-2" in ids

    def test_default_model(self, provider):
        assert provider.default_model() == "gpt-image-1.5"


class TestAvailability:
    def test_no_api_key_unavailable(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert openai_plugin.OpenAIImageGenProvider().is_available() is False

    def test_api_key_set_available(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "test")
        assert openai_plugin.OpenAIImageGenProvider().is_available() is True


class TestGenerate:
    def test_empty_prompt_rejected(self, provider):
        result = provider.generate("", aspect_ratio="square")
        assert result["success"] is False
        assert result["error_type"] == "invalid_argument"

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = openai_plugin.OpenAIImageGenProvider().generate("a cat")
        assert result["success"] is False
        assert result["error_type"] == "auth_required"

    def test_gpt_image_b64_saves_to_cache(self, provider, tmp_path, monkeypatch):
        # 1x1 PNG in base64 (valid image bytes)
        import base64
        png_bytes = bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
            "890000000d49444154789c6300010000000500010d0a2db40000000049454e44"
            "ae426082"
        )
        b64 = base64.b64encode(png_bytes).decode()

        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(b64=b64)

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client
        with patch.dict("sys.modules", {"openai": fake_openai}):
            result = provider.generate("a cat", aspect_ratio="landscape")

        assert result["success"] is True
        assert result["model"] == "gpt-image-1.5"
        assert result["aspect_ratio"] == "landscape"
        assert result["provider"] == "openai"
        # Saved to HERMES_HOME/cache/images/
        saved = Path(result["image"])
        assert saved.exists()
        assert saved.parent == tmp_path / "cache" / "images"
        assert saved.read_bytes() == png_bytes

        # Request payload: landscape → 1536x1024, response_format=b64_json
        call_kwargs = fake_client.images.generate.call_args.kwargs
        assert call_kwargs["size"] == "1536x1024"
        assert call_kwargs["response_format"] == "b64_json"
        assert call_kwargs["model"] == "gpt-image-1.5"

    def test_dalle3_url_passthrough(self, provider, tmp_path, monkeypatch):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump(
                {"image_gen": {"openai": {"model": "dall-e-3", "quality": "hd"}}}
            )
        )

        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            url="https://example.com/img.png",
            revised_prompt="A photo of a cat",
        )

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client
        with patch.dict("sys.modules", {"openai": fake_openai}):
            result = provider.generate("a cat", aspect_ratio="portrait")

        assert result["success"] is True
        assert result["model"] == "dall-e-3"
        assert result["image"] == "https://example.com/img.png"
        assert result["revised_prompt"] == "A photo of a cat"

        call_kwargs = fake_client.images.generate.call_args.kwargs
        assert call_kwargs["size"] == "1024x1792"  # portrait on dalle3
        assert call_kwargs["response_format"] == "url"
        assert call_kwargs["quality"] == "hd"

    def test_dalle2_clamps_to_square(self, provider, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"model": "dall-e-2"}}})
        )

        fake_client = MagicMock()
        fake_client.images.generate.return_value = _fake_response(
            url="https://example.com/x.png"
        )

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client
        with patch.dict("sys.modules", {"openai": fake_openai}):
            # Every aspect_ratio clamps to 1024x1024 for dalle2
            for ar in ("landscape", "square", "portrait"):
                provider.generate("a cat", aspect_ratio=ar)

        for call in fake_client.images.generate.call_args_list:
            assert call.kwargs["size"] in {"1024x1024", "512x512", "256x256"}

    def test_api_error_returns_error_response(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.side_effect = RuntimeError("boom")

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client
        with patch.dict("sys.modules", {"openai": fake_openai}):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "api_error"
        assert "boom" in result["error"]

    def test_empty_response_data(self, provider):
        fake_client = MagicMock()
        fake_client.images.generate.return_value = SimpleNamespace(data=[])

        fake_openai = MagicMock()
        fake_openai.OpenAI.return_value = fake_client
        with patch.dict("sys.modules", {"openai": fake_openai}):
            result = provider.generate("a cat")

        assert result["success"] is False
        assert result["error_type"] == "empty_response"


class TestModelResolution:
    def test_env_var_override_wins(self, provider, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "dall-e-3")
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "dall-e-3"
        assert meta["family"] == "dalle3"

    def test_env_var_unknown_ignored(self, provider, monkeypatch):
        monkeypatch.setenv("OPENAI_IMAGE_MODEL", "some-bogus-model")
        model_id, _ = openai_plugin._resolve_model()
        # Falls through to default
        assert model_id == openai_plugin.DEFAULT_OPENAI_MODEL

    def test_config_openai_model(self, provider, tmp_path):
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.safe_dump({"image_gen": {"openai": {"model": "dall-e-2"}}})
        )
        model_id, meta = openai_plugin._resolve_model()
        assert model_id == "dall-e-2"
        assert meta["family"] == "dalle2"
