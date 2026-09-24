"""Request/response schemas for the versioned deconvolution API."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class CoelutingRequest(BaseModel):
    """Request for joint multi-charge coelution confirmation.

    Exactly one mutually disjoint cluster must be chosen for every required
    charge state, and the neutral-mass tolerance intervals derived from the
    clusters' first peaks must share a common intersection.
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
    required_charges: list[StrictPositiveInt] = Field(
        min_length=2,
        max_length=4,
        description=(
            "2 to 4 charge states that must each contribute exactly one "
            "cluster; positive, unique, and every entry must be in charges."
        ),
    )
    tolerance: Decimal = Field(
        description="Non-negative decimal m/z tolerance applied to 1.003355/z."
    )
    mass_tolerance: Decimal = Field(
        description=(
            "Non-negative decimal neutral-mass tolerance (Da); a cluster at "
            "charge z with first peak m/z m witnesses M = z*m within "
            "[M - mass_tolerance, M + mass_tolerance]."
        ),
    )

    @field_validator("tolerance", "mass_tolerance")
    @classmethod
    def _tolerances_valid(cls, value: Decimal) -> Decimal:
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

    @field_validator("required_charges")
    @classmethod
    def _required_charges_valid(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError(
                "required_charges must be a set: duplicate values are not allowed"
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

    @field_validator("required_charges")
    @classmethod
    def _required_charges_subset(cls, value: list[int], info) -> list[int]:
        charges = info.data.get("charges")
        if charges is not None:
            allowed = set(charges)
            extra = [z for z in value if z not in allowed]
            if extra:
                raise ValueError(
                    "every required charge must be in charges: "
                    f"{sorted(extra)} not allowed"
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


# ---------------------------------------------------------------------- #
# Coeluting endpoint responses
# ---------------------------------------------------------------------- #


class MassIntervalOut(BaseModel):
    """Inclusive neutral-mass interval [lower, upper] in Dalton."""

    lower: str
    upper: str


class CoelutingClusterOut(ClusterOut):
    #: Neutral mass derived from the first peak: charge * first-peak m/z.
    neutral_mass: str


class CoelutingSolutionOut(BaseModel):
    clusters: list[CoelutingClusterOut]
    unexplained_peaks: list[PeakOut]
    common_mass: MassIntervalOut


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
    clusters: list[CoelutingClusterOut]
    unexplained_peaks: list[PeakOut]
    common_mass: MassIntervalOut | None
    second_witness: CoelutingSolutionOut | None
    input_summary: CoelutingInputSummaryOut
