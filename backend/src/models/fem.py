"""FEM simulation request and response models."""

from typing import Any

from pydantic import Field, field_validator, model_validator

from src.models.base import StrictModel
from src.models.limits import MAX_FREQUENCY_MHZ, MIN_FREQUENCY_MHZ


class FEMFrequencyConfig(StrictModel):
    """Frequency sweep for a FEM job."""

    start_mhz: float = Field(ge=MIN_FREQUENCY_MHZ, le=MAX_FREQUENCY_MHZ)
    stop_mhz: float = Field(ge=MIN_FREQUENCY_MHZ, le=MAX_FREQUENCY_MHZ)
    steps: int = Field(ge=2, le=201)

    @model_validator(mode="after")
    def validate_range(self) -> "FEMFrequencyConfig":
        if self.stop_mhz < self.start_mhz:
            raise ValueError("stop_mhz must be >= start_mhz")
        return self


class FEMSimulationRequest(StrictModel):
    """ModelIR and numerical settings for a FEM frequency sweep."""

    model_ir: dict[str, Any]
    frequency: FEMFrequencyConfig
    port_id: str | None = None
    mesh_size_m: float = Field(default=0.005, gt=0.0, le=1.0)
    reference_impedance_ohms: float = Field(default=50.0, gt=0.0, le=10000.0)

    @field_validator("model_ir")
    @classmethod
    def model_ir_must_be_object(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("model_ir must not be empty")
        return value


class FEMFrequencyResult(StrictModel):
    frequency_mhz: float
    impedance_real_ohms: float
    impedance_imag_ohms: float
    s11_real: float
    s11_imag: float
    s11_db: float


class FEMMeshInfo(StrictModel):
    elements: int
    nodes: int
    length_m: float
    warnings: list[str] = Field(default_factory=list)


class FEMSimulationResult(StrictModel):
    simulation_id: str
    engine: str = "fem-reduced"
    computed_in_ms: float
    mesh: FEMMeshInfo
    frequency_data: list[FEMFrequencyResult]
    resonant_frequency_mhz: float | None = None
    warnings: list[str] = Field(default_factory=list)
