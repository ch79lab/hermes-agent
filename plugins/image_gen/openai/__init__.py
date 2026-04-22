"""OpenAI image generation backend.

Exposes the ``openai`` image generation API as an :class:`ImageGenProvider`
implementation. Supports four model families:

* ``gpt-image-1.5`` (default) — newest, highest quality
* ``gpt-image-1`` — prior generation
* ``dall-e-3`` — legacy DALL-E 3
* ``dall-e-2`` — legacy DALL-E 2 (only 256/512/1024 squares)

Outputs:

* gpt-image-* → base64 JSON → saved under ``$HERMES_HOME/cache/images/``
* dall-e-3 → URL passthrough
* dall-e-2 → URL passthrough

Config overrides live at ``image_gen.openai.*`` in ``config.yaml``; the
``quality`` knob is honored for gpt-image / dalle3 families.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.image_gen_provider import (
    DEFAULT_ASPECT_RATIO,
    ImageGenProvider,
    error_response,
    resolve_aspect_ratio,
    save_b64_image,
    success_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model catalog
# ---------------------------------------------------------------------------
#
# ``family`` determines how aspect_ratio maps to a ``size`` string and which
# request parameters the API accepts. Three families:
#
#   "gpt_image"  — gpt-image-1.5 / gpt-image-1; supports 1024x1024 /
#                  1536x1024 / 1024x1536; returns b64 by default.
#   "dalle3"     — dall-e-3; supports 1024x1024 / 1792x1024 / 1024x1792;
#                  returns URL.
#   "dalle2"     — dall-e-2; squares only (256/512/1024); returns URL.

_OPENAI_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-image-1.5": {
        "display": "GPT Image 1.5",
        "speed": "~10s",
        "strengths": "Highest quality, strong prompt adherence, text rendering",
        "price": "varies",
        "family": "gpt_image",
        "sizes": {
            "landscape": "1536x1024",
            "square": "1024x1024",
            "portrait": "1024x1536",
        },
        "default_quality": "auto",
    },
    "gpt-image-1": {
        "display": "GPT Image 1",
        "speed": "~10s",
        "strengths": "Prior-gen GPT image",
        "price": "varies",
        "family": "gpt_image",
        "sizes": {
            "landscape": "1536x1024",
            "square": "1024x1024",
            "portrait": "1024x1536",
        },
        "default_quality": "auto",
    },
    "gpt-image-1-mini": {
        "display": "GPT Image 1 Mini",
        "speed": "~5s",
        "strengths": "Faster, cheaper GPT image",
        "price": "varies",
        "family": "gpt_image",
        "sizes": {
            "landscape": "1536x1024",
            "square": "1024x1024",
            "portrait": "1024x1536",
        },
        "default_quality": "auto",
    },
    "dall-e-3": {
        "display": "DALL·E 3",
        "speed": "~8s",
        "strengths": "Legacy DALL-E 3",
        "price": "$0.04-0.12 / image",
        "family": "dalle3",
        "sizes": {
            "landscape": "1792x1024",
            "square": "1024x1024",
            "portrait": "1024x1792",
        },
        "default_quality": "standard",
    },
    "dall-e-2": {
        "display": "DALL·E 2",
        "speed": "~5s",
        "strengths": "Legacy DALL-E 2 (squares only)",
        "price": "$0.016-0.020 / image",
        "family": "dalle2",
        "sizes": {
            "landscape": "1024x1024",
            "square": "1024x1024",
            "portrait": "1024x1024",
        },
        "default_quality": None,
    },
}

DEFAULT_OPENAI_MODEL = "gpt-image-1.5"


def _load_openai_config() -> Dict[str, Any]:
    """Read ``image_gen`` and ``image_gen.openai`` from config.yaml."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        section = cfg.get("image_gen") if isinstance(cfg, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:
        logger.debug("Could not load image_gen config: %s", exc)
        return {}


def _resolve_model() -> Tuple[str, Dict[str, Any]]:
    """Decide which OpenAI model to use and return ``(model_id, meta)``.

    Lookup order:

    1. ``OPENAI_IMAGE_MODEL`` env var (escape hatch for scripts/tests)
    2. ``image_gen.openai.model`` from config.yaml
    3. ``image_gen.model`` if it starts with ``gpt-image`` / ``dall-e``
    4. :data:`DEFAULT_OPENAI_MODEL`
    """
    env_override = os.environ.get("OPENAI_IMAGE_MODEL")
    if env_override and env_override in _OPENAI_MODELS:
        return env_override, _OPENAI_MODELS[env_override]

    cfg = _load_openai_config()
    openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
    candidate = openai_cfg.get("model") if isinstance(openai_cfg, dict) else None
    if not candidate:
        top = cfg.get("model")
        if isinstance(top, str) and top in _OPENAI_MODELS:
            candidate = top

    if isinstance(candidate, str) and candidate in _OPENAI_MODELS:
        return candidate, _OPENAI_MODELS[candidate]

    return DEFAULT_OPENAI_MODEL, _OPENAI_MODELS[DEFAULT_OPENAI_MODEL]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class OpenAIImageGenProvider(ImageGenProvider):
    """OpenAI images.generate backend."""

    @property
    def name(self) -> str:
        return "openai"

    @property
    def display_name(self) -> str:
        return "OpenAI"

    def is_available(self) -> bool:
        if not os.environ.get("OPENAI_API_KEY"):
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        """Return catalog entries for the model picker."""
        out: List[Dict[str, Any]] = []
        for model_id, meta in _OPENAI_MODELS.items():
            out.append(
                {
                    "id": model_id,
                    "display": meta.get("display", model_id),
                    "speed": meta.get("speed", ""),
                    "strengths": meta.get("strengths", ""),
                    "price": meta.get("price", ""),
                }
            )
        return out

    def default_model(self) -> Optional[str]:
        return DEFAULT_OPENAI_MODEL

    def generate(
        self,
        prompt: str,
        aspect_ratio: str = DEFAULT_ASPECT_RATIO,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        prompt = (prompt or "").strip()
        aspect = resolve_aspect_ratio(aspect_ratio)

        if not prompt:
            return error_response(
                error="Prompt is required and must be a non-empty string",
                error_type="invalid_argument",
                provider="openai",
                aspect_ratio=aspect,
            )

        if not os.environ.get("OPENAI_API_KEY"):
            return error_response(
                error=(
                    "OPENAI_API_KEY not set. Run `hermes tools` → Image "
                    "Generation → OpenAI to configure, or `hermes setup` "
                    "to add the key."
                ),
                error_type="auth_required",
                provider="openai",
                aspect_ratio=aspect,
            )

        try:
            import openai
        except ImportError:
            return error_response(
                error="openai Python package not installed (pip install openai)",
                error_type="missing_dependency",
                provider="openai",
                aspect_ratio=aspect,
            )

        model_id, meta = _resolve_model()
        family = meta["family"]
        size = meta["sizes"].get(aspect, meta["sizes"]["square"])

        payload: Dict[str, Any] = {
            "model": model_id,
            "prompt": prompt,
            "size": size,
            "n": 1,
        }

        cfg = _load_openai_config()
        openai_cfg = cfg.get("openai") if isinstance(cfg.get("openai"), dict) else {}
        if not isinstance(openai_cfg, dict):
            openai_cfg = {}

        if family == "gpt_image":
            # gpt-image-* returns b64_json unconditionally and REJECTS
            # ``response_format`` as an unknown parameter (verified live
            # April 2026: 400 invalid_request_error). Don't send it.
            quality = openai_cfg.get("quality") or meta.get("default_quality")
            if quality and quality != "auto":
                payload["quality"] = quality
        elif family == "dalle3":
            payload["response_format"] = "url"
            quality = openai_cfg.get("quality") or meta.get("default_quality")
            if quality and quality in {"standard", "hd"}:
                payload["quality"] = quality
        elif family == "dalle2":
            payload["response_format"] = "url"
            # dalle2 only accepts 256x256 / 512x512 / 1024x1024. Our sizes
            # table already clamps to 1024x1024 but surface a clear error
            # if someone overrode it out-of-range.
            if payload["size"] not in {"256x256", "512x512", "1024x1024"}:
                payload["size"] = "1024x1024"

        try:
            client = openai.OpenAI()
            response = client.images.generate(**payload)
        except Exception as exc:
            logger.debug("OpenAI image generation failed", exc_info=True)
            return error_response(
                error=f"OpenAI image generation failed: {exc}",
                error_type="api_error",
                provider="openai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        data = getattr(response, "data", None) or []
        if not data:
            return error_response(
                error="OpenAI returned no image data",
                error_type="empty_response",
                provider="openai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        first = data[0]
        url = getattr(first, "url", None)
        b64 = getattr(first, "b64_json", None)
        revised_prompt = getattr(first, "revised_prompt", None)

        if b64:
            try:
                saved_path = save_b64_image(
                    b64, prefix=f"openai_{model_id.replace('.', '-')}"
                )
            except Exception as exc:
                return error_response(
                    error=f"Could not save image to cache: {exc}",
                    error_type="io_error",
                    provider="openai",
                    model=model_id,
                    prompt=prompt,
                    aspect_ratio=aspect,
                )
            image_ref = str(saved_path)
        elif url:
            image_ref = url
        else:
            return error_response(
                error="OpenAI response contained neither URL nor b64_json",
                error_type="empty_response",
                provider="openai",
                model=model_id,
                prompt=prompt,
                aspect_ratio=aspect,
            )

        extra: Dict[str, Any] = {"size": size}
        if revised_prompt:
            extra["revised_prompt"] = revised_prompt

        return success_response(
            image=image_ref,
            model=model_id,
            prompt=prompt,
            aspect_ratio=aspect,
            provider="openai",
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx) -> None:
    """Plugin entry point — wire ``OpenAIImageGenProvider`` into the registry."""
    ctx.register_image_gen_provider(OpenAIImageGenProvider())
