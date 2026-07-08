from __future__ import annotations

from importlib import import_module
from typing import Any, Mapping

from loguru import logger

from skyrl.train.dataset.modalities import ModalityHandlerSpec


def _resolve_target(target: str):
    if ":" in target:
        module_path, attr = target.split(":", 1)
    elif "." in target:
        module_path, attr = target.rsplit(".", 1)
    else:
        raise ValueError(
            f"Handler target `{target}` is invalid. Expected format `pkg.module:factory` or `pkg.module.Factory`."
        )

    module = import_module(module_path)
    try:
        handler = getattr(module, attr)
    except AttributeError as exc:
        raise ValueError(f"Module `{module_path}` has no attribute `{attr}` referenced by `{target}`.") from exc
    return handler


def instantiate_handler(
    handler_spec: ModalityHandlerSpec,
    *,
    modality_id: str,
    role: str,
    extra_kwargs: Mapping[str, Any] | None = None,
) -> Any:
    factory = _resolve_target(handler_spec.target)
    kwargs = dict(handler_spec.kwargs)
    kwargs.setdefault("modality_id", modality_id)
    kwargs.setdefault("role", role)
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    try:
        instance = factory(**kwargs)
    except TypeError:
        logger.debug(
            "Falling back to calling `%s` without keyword arguments. kwargs=%s",
            handler_spec.target,
            kwargs,
        )
        instance = factory()

    logger.info(
        "Initialized %s handler for modality `%s` using `%s`.",
        role,
        modality_id,
        handler_spec.target,
    )
    return instance


__all__ = ["instantiate_handler"]
