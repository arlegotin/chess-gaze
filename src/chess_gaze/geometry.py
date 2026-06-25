from __future__ import annotations

import math
from enum import Enum, StrEnum
from typing import Annotated, Any, get_args, get_origin

from pydantic import AfterValidator, BaseModel, ConfigDict, model_validator


class CoordinateSpace(StrEnum):
    IMAGE_PX = "IMAGE_PX"
    NORMALIZED = "NORMALIZED"


def validate_finite_float(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return value


RotationRadians = Annotated[float, AfterValidator(validate_finite_float)]


def enum_type_from_annotation(annotation: Any) -> type[Enum] | None:
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation

    origin = get_origin(annotation)
    if origin is None:
        return None

    for candidate in get_args(annotation):
        enum_type = enum_type_from_annotation(candidate)
        if enum_type is not None:
            return enum_type

    return None


class StrictSchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @model_validator(mode="before")
    @classmethod
    def coerce_enum_strings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        coerced = dict(data)
        for field_name, model_field in cls.model_fields.items():
            if field_name not in coerced:
                continue

            enum_type = enum_type_from_annotation(model_field.annotation)
            if enum_type is None or not isinstance(coerced[field_name], str):
                continue

            try:
                coerced[field_name] = enum_type(coerced[field_name])
            except ValueError:
                continue

        return coerced

    @model_validator(mode="after")
    def reject_non_finite_floats(self) -> StrictSchemaModel:
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        return self


class Point2D(StrictSchemaModel):
    space: CoordinateSpace
    x: float
    y: float


class BBox(StrictSchemaModel):
    space: CoordinateSpace
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @model_validator(mode="after")
    def validate_bounds(self) -> BBox:
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")
        return self


class Transform2D(StrictSchemaModel):
    source_space: CoordinateSpace
    target_space: CoordinateSpace
    m00: float
    m01: float
    m02: float
    m10: float
    m11: float
    m12: float
