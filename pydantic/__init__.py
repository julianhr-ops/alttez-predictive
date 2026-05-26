from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable, Literal, Union, get_args, get_origin, get_type_hints
import types


class ValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def ConfigDict(**kwargs: Any) -> dict[str, Any]:
    return dict(kwargs)


_MISSING = object()


@dataclass(slots=True)
class FieldInfo:
    default: Any = _MISSING
    default_factory: Callable[[], Any] | None = None
    alias: str | None = None
    ge: float | None = None
    le: float | None = None


def Field(
    default: Any = _MISSING,
    *,
    default_factory: Callable[[], Any] | None = None,
    alias: str | None = None,
    ge: float | None = None,
    le: float | None = None,
) -> FieldInfo:
    return FieldInfo(
        default=default,
        default_factory=default_factory,
        alias=alias,
        ge=ge,
        le=le,
    )


def field_validator(*field_names: str, mode: str = "after"):
    def decorator(fn):
        setattr(fn, "__field_validator_fields__", field_names)
        setattr(fn, "__field_validator_mode__", mode)
        return fn

    return decorator


def model_validator(*, mode: str = "after"):
    def decorator(fn):
        setattr(fn, "__model_validator_mode__", mode)
        return fn

    return decorator


def _strip_z(value: str) -> str:
    return value[:-1] + "+00:00" if value.endswith("Z") else value


def _coerce_primitive(annotation: Any, value: Any) -> Any:
    if value is None:
        return None

    if annotation is Any:
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Literal:
        if value not in args:
            raise ValueError(f"value {value!r} is not one of {args!r}")
        return value

    if origin in {list, tuple}:
        item_type = args[0] if args else Any
        if not isinstance(value, (list, tuple)):
            raise ValueError("expected a list or tuple")
        return [_coerce_value(item_type, item) for item in value]

    if origin is dict:
        if not isinstance(value, dict):
            raise ValueError("expected a dictionary")
        return value

    if origin in {Union, types.UnionType}:
        non_none_args = [item for item in args if item is not type(None)]
        if value is None:
            return None
        last_error: Exception | None = None
        for item_type in non_none_args:
            try:
                return _coerce_value(item_type, value)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        if last_error is not None:
            raise last_error
        return value

    if isinstance(annotation, type):
        if issubclass(annotation, BaseModel):
            if isinstance(value, annotation):
                return value
            if isinstance(value, dict):
                return annotation.model_validate(value)
            raise ValueError(f"expected {annotation.__name__} or mapping")

        if issubclass(annotation, Enum):
            return annotation(value)

        if annotation is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "y"}:
                    return True
                if normalized in {"false", "0", "no", "n"}:
                    return False
            return bool(value)

        if annotation is int:
            if isinstance(value, int) and not isinstance(value, bool):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                return int(float(value.strip().replace(",", ".")))

        if annotation is float:
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
            if isinstance(value, str):
                return float(value.strip().replace("%", "").replace(",", "."))

        if annotation is str:
            if value is None:
                raise ValueError("value is required")
            return str(value).strip()

        if annotation is date:
            if isinstance(value, date) and not isinstance(value, datetime):
                return value
            if isinstance(value, str):
                return date.fromisoformat(value.strip())

        if annotation is datetime:
            if isinstance(value, datetime):
                return value
            if isinstance(value, str):
                return datetime.fromisoformat(_strip_z(value.strip()))

    return value


def _coerce_value(annotation: Any, value: Any) -> Any:
    if value is None:
        origin = get_origin(annotation)
        args = get_args(annotation)
        if origin in {list, dict}:
            return None
        if annotation is Any:
            return None
        if origin in {Union, types.UnionType} and type(None) in args:
            return None
        if annotation is type(None):
            return None
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            raise ValueError("value is required")
        return None

    return _coerce_primitive(annotation, value)


class BaseModelMeta(type):
    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        cls.__field_info__ = {}
        cls.__field_validators__ = {}
        cls.__model_validators__ = []

        for base in reversed(bases):
            cls.__field_info__.update(getattr(base, "__field_info__", {}))
            for field_name, validators in getattr(base, "__field_validators__", {}).items():
                cls.__field_validators__.setdefault(field_name, []).extend(validators)
            cls.__model_validators__.extend(getattr(base, "__model_validators__", []))

        annotations = namespace.get("__annotations__", {})
        for field_name in annotations:
            field_value = namespace.get(field_name, _MISSING)
            if isinstance(field_value, FieldInfo):
                cls.__field_info__[field_name] = field_value
            elif field_value is not _MISSING:
                cls.__field_info__[field_name] = FieldInfo(default=field_value)
            else:
                cls.__field_info__.setdefault(field_name, FieldInfo())

        for attr_name, attr_value in namespace.items():
            candidates = [attr_value]
            if isinstance(attr_value, (classmethod, staticmethod)):
                candidates.append(attr_value.__func__)
            for candidate in candidates:
                if hasattr(candidate, "__field_validator_fields__"):
                    for field_name in getattr(candidate, "__field_validator_fields__"):
                        cls.__field_validators__.setdefault(field_name, []).append(
                            (getattr(candidate, "__field_validator_mode__", "after"), attr_name)
                        )
                if hasattr(candidate, "__model_validator_mode__"):
                    cls.__model_validators__.append((getattr(candidate, "__model_validator_mode__", "after"), attr_name))

        return cls


class BaseModel(metaclass=BaseModelMeta):
    model_config: dict[str, Any] = {}

    def __init__(self, **data: Any) -> None:
        validated = self.__class__.model_validate(data)
        self.__dict__.update(validated.__dict__)

    @classmethod
    def model_validate(cls, data: Any) -> "BaseModel":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            raise ValidationError([f"{cls.__name__} requires a mapping for validation"])

        errors: list[str] = []
        values: dict[str, Any] = {}
        type_hints = get_type_hints(cls, include_extras=True)
        config = getattr(cls, "model_config", {}) or {}
        extra_mode = config.get("extra", "ignore")
        populate_by_name = bool(config.get("populate_by_name", False))
        remaining = dict(data)

        for field_name, annotation in type_hints.items():
            if field_name.startswith("__"):
                continue
            field_info = cls.__field_info__.get(field_name, FieldInfo())
            raw_value = _MISSING
            if field_info.alias and field_info.alias in remaining:
                raw_value = remaining.pop(field_info.alias)
            elif field_name in remaining:
                raw_value = remaining.pop(field_name)
            elif populate_by_name and field_info.alias and field_name in data:
                raw_value = data[field_name]

            if raw_value is _MISSING:
                if field_info.default_factory is not None:
                    raw_value = field_info.default_factory()
                elif field_info.default is not _MISSING:
                    raw_value = field_info.default
                else:
                    errors.append(f"Missing required field: {field_name}")
                    continue

            validators = cls.__field_validators__.get(field_name, [])
            for mode, validator_name in validators:
                if mode == "before":
                    validator = getattr(cls, validator_name)
                    raw_value = validator(raw_value)

            try:
                value = _coerce_value(annotation, raw_value)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{field_name}: {exc}")
                continue

            if value is not None:
                if field_info.ge is not None and value < field_info.ge:
                    errors.append(f"{field_name} must be >= {field_info.ge}")
                    continue
                if field_info.le is not None and value > field_info.le:
                    errors.append(f"{field_name} must be <= {field_info.le}")
                    continue

            for mode, validator_name in validators:
                if mode != "before":
                    validator = getattr(cls, validator_name)
                    value = validator(value)

            values[field_name] = value

        if extra_mode == "forbid" and remaining:
            errors.append(f"Unexpected fields: {', '.join(sorted(remaining))}")

        if errors:
            raise ValidationError(errors)

        instance = cls.__new__(cls)
        instance.__dict__.update(values)

        for mode, validator_name in cls.__model_validators__:
            if mode == "after":
                validator = getattr(instance, validator_name)
                result = validator()
                if result is not None:
                    instance = result

        return instance

    def model_dump(self, *, by_alias: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {}
        type_hints = get_type_hints(self.__class__, include_extras=True)
        for field_name in type_hints:
            if field_name.startswith("__"):
                continue
            field_info = self.__class__.__field_info__.get(field_name, FieldInfo())
            key = field_info.alias if by_alias and field_info.alias else field_name
            value = getattr(self, field_name)
            if isinstance(value, BaseModel):
                data[key] = value.model_dump(by_alias=by_alias)
            else:
                data[key] = value
        return data

    def __repr__(self) -> str:
        fields = ", ".join(f"{key}={value!r}" for key, value in self.model_dump().items())
        return f"{self.__class__.__name__}({fields})"
