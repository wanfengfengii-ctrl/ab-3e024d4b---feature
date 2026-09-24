"""Request/response schemas for the versioned deconvolution API."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

# Strict: JSON floats such as 1.5 or numeric strings must not be accepted
# where a positive integer is required.
StrictPositiveInt = Annotated[int, Field(strict=True, gt=0)]


class PeakInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mz: Decimal = Field(description="Mass-to-charge ratio; finite decimal > 0.")
    intensity: StrictPositiveInt = Field(description="Positive integer intensity.")

    @field_validator("mz")
    @classmethod
    def _mz_finite_positive(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("mz must be a finite decimal number")
        if value <= 0:
            raise ValueError("mz must be greater than 0")
        return value


class DeconvolutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    peaks: list[PeakInput] = Field(
        min_length=2,
        max_length=36,
        description="2 to 36 peaks, strictly increasing in mz.",
    )
    charges: list[StrictPositiveInt] = Field(
        min_length=1,
        description="Allowed charge states (a set: positive, unique integers).",
    )
    tolerance: Decimal = Field(
        description="Non-negative decimal m/z tolerance applied to 1.003355/z."
    )

    @field_validator("tolerance")
    @classmethod
    def _tolerance_valid(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("tolerance must be a finite decimal number")
        if value < 0:
            raise ValueError("tolerance must be greater than or equal to 0")
        return value

    @field_validator("charges")
    @classmethod
    def _charges_unique(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("charges must be a set: duplicate values are not allowed")
        return value

    @field_validator("peaks")
    @classmethod
    def _peaks_strictly_increasing(cls, value: list[PeakInput]) -> list[PeakInput]:
        for i in range(1, len(value)):
            if value[i].mz <= value[i - 1].mz:
                raise ValueError(
                    "peaks must be strictly increasing in mz: "
                    f"peaks[{i}].mz={value[i].mz} is not greater than "
                    f"peaks[{i - 1}].mz={value[i - 1].mz}"
                )
        return value


# ---------------------------------------------------------------------- #
# Co-eluting multi-charge confirmation
# ---------------------------------------------------------------------- #


class CoelutingRequest(BaseModel):
    """Confirm one precursor's envelope across 2-4 charge states.

    Reuses the peak list, allowed charge set and inter-peak tolerance of the
    plain deconvolution endpoint, and additionally requires the charge states
    that must appear simultaneously plus a neutral-mass tolerance.
    """

    model_config = ConfigDict(extra="forbid")

    peaks: list[PeakInput] = Field(
        min_length=2,
        max_length=36,
        description="2 to 36 peaks, strictly increasing in mz.",
    )
    charges: list[StrictPositiveInt] = Field(
        min_length=1,
        description="Allowed charge states (a set: positive, unique integers).",
    )
    tolerance: Decimal = Field(
        description="Non-negative decimal m/z tolerance applied to 1.003355/z."
    )
    required_charges: list[StrictPositiveInt] = Field(
        min_length=2,
        max_length=4,
        description=(
            "2 to 4 charge states that must co-occur: exactly one cluster is "
            "selected per required charge (a set: positive, unique, and a "
            "subset of charges)."
        ),
    )
    mass_tolerance: Decimal = Field(
        description=(
            "Non-negative decimal tolerance on the neutral-mass estimate "
            "(first-peak m/z x charge); the per-cluster intervals "
            "[M - mass_tolerance, M + mass_tolerance] must intersect."
        )
    )

    @field_validator("tolerance")
    @classmethod
    def _tolerance_valid(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("tolerance must be a finite decimal number")
        if value < 0:
            raise ValueError("tolerance must be greater than or equal to 0")
        return value

    @field_validator("mass_tolerance")
    @classmethod
    def _mass_tolerance_valid(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("mass_tolerance must be a finite decimal number")
        if value < 0:
            raise ValueError("mass_tolerance must be greater than or equal to 0")
        return value

    @field_validator("charges")
    @classmethod
    def _charges_unique(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("charges must be a set: duplicate values are not allowed")
        return value

    @field_validator("required_charges")
    @classmethod
    def _required_charges_unique(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError(
                "required_charges must be a set: duplicate values are not allowed"
            )
        return value

    @field_validator("required_charges")
    @classmethod
    def _required_charges_subset(
        cls, value: list[int], info: ValidationInfo
    ) -> list[int]:
        charges = info.data.get("charges")
        if charges is not None:
            unknown = [z for z in value if z not in set(charges)]
            if unknown:
                raise ValueError(
                    "required_charges must be a subset of charges; "
                    f"not allowed: {unknown}"
                )
        return value

    @field_validator("peaks")
    @classmethod
    def _peaks_strictly_increasing(cls, value: list[PeakInput]) -> list[PeakInput]:
        for i in range(1, len(value)):
            if value[i].mz <= value[i - 1].mz:
                raise ValueError(
                    "peaks must be strictly increasing in mz: "
                    f"peaks[{i}].mz={value[i].mz} is not greater than "
                    f"peaks[{i - 1}].mz={value[i - 1].mz}"
                )
        return value


# ---------------------------------------------------------------------- #
# Responses
# ---------------------------------------------------------------------- #


class PeakOut(BaseModel):
    index: int
    mz: str
    intensity: int


class ClusterOut(BaseModel):
    charge: int
    peak_indices: list[int]
    explained_intensity: int
    peaks: list[PeakOut]


class SolutionOut(BaseModel):
    clusters: list[ClusterOut]
    unexplained_peaks: list[PeakOut]


class ObjectivesOut(BaseModel):
    explained_intensity: int
    explained_peak_count: int
    cluster_count: int


class InputSummaryOut(BaseModel):
    peak_count: int
    charges: list[int]
    tolerance: str
    isotope_spacing: str


class DeconvolutionResponse(BaseModel):
    verdict: Literal["UNIQUE", "AMBIGUOUS", "UNRESOLVED"]
    objectives: ObjectivesOut
    clusters: list[ClusterOut]
    unexplained_peaks: list[PeakOut]
    second_witness: SolutionOut | None
    input_summary: InputSummaryOut


class MassIntervalOut(BaseModel):
    """Common intersection of the combination's neutral-mass intervals."""

    lower: str
    upper: str


class CoelutingSolutionOut(BaseModel):
    clusters: list[ClusterOut]
    common_mass_interval: MassIntervalOut
    unexplained_peaks: list[PeakOut]


class CoelutingInputSummaryOut(BaseModel):
    peak_count: int
    charges: list[int]
    required_charges: list[int]
    tolerance: str
    mass_tolerance: str
    isotope_spacing: str


class CoelutingResponse(BaseModel):
    verdict: Literal["UNIQUE", "AMBIGUOUS", "UNRESOLVED"]
    objectives: ObjectivesOut
    clusters: list[ClusterOut]
    common_mass_interval: MassIntervalOut | None
    unexplained_peaks: list[PeakOut]
    second_witness: CoelutingSolutionOut | None
    input_summary: CoelutingInputSummaryOut
