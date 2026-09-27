"""Dynamic Domain and Task Registries for ZorQelis VerQen-T1.

Supports:
- Dynamic discovery of capability domains (Sections 15 & 16)
- Independent execution of task generators (Section 17)
- Compositional multi-domain registration
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Type


class DomainRegistry:
    """Registry for capability domains, preventing brittle if/elif trees."""
    _domains: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, domain_meta: Optional[Dict[str, Any]] = None) -> Callable:
        """Decorator to register a domain generator class or specification."""
        def decorator(subclass: Any) -> Any:
            cls._domains[name] = {
                "class": subclass,
                "metadata": domain_meta or {},
            }
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        """Retrieve registered domain class."""
        entry = cls._domains.get(name)
        return entry["class"] if entry else None

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """Retrieve domain metadata."""
        entry = cls._domains.get(name)
        return entry["metadata"] if entry else {}

    @classmethod
    def list_domains(cls) -> List[str]:
        """Return list of all registered domain names."""
        return sorted(list(cls._domains.keys()))

    @classmethod
    def clear(cls) -> None:
        """Clear registry (useful for testing)."""
        cls._domains.clear()


class TaskRegistry:
    """Registry for individual task generators."""
    _tasks: Dict[str, Any] = {}

    @classmethod
    def register(cls, task_key: str, task_family: str, domain: str) -> Callable:
        """Decorator to register a specific task generator callable."""
        def decorator(fn_or_cls: Any) -> Any:
            cls._tasks[task_key] = {
                "handler": fn_or_cls,
                "task_family": task_family,
                "domain": domain,
            }
            return fn_or_cls
        return decorator

    @classmethod
    def get(cls, task_key: str) -> Optional[Any]:
        """Retrieve task generator."""
        entry = cls._tasks.get(task_key)
        return entry["handler"] if entry else None

    @classmethod
    def list_tasks(cls, domain: Optional[str] = None) -> List[str]:
        """List registered task keys, optionally filtered by domain."""
        if domain:
            return sorted([k for k, v in cls._tasks.items() if v["domain"] == domain])
        return sorted(list(cls._tasks.keys()))
