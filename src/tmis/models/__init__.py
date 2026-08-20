from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model import CoreOutput, SelectorGuidedMultiPathModel

__all__ = [
    "SelectorGuidedMultiPathModel",
    "CoreOutput",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .model import CoreOutput, SelectorGuidedMultiPathModel

        return {
            "SelectorGuidedMultiPathModel": SelectorGuidedMultiPathModel,
            "CoreOutput": CoreOutput,
        }[name]
    raise AttributeError(name)
