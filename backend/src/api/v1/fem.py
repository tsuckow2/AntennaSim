"""POST /api/v1/fem/simulate — Run the ModelIR FEM frequency sweep."""

from fastapi import APIRouter, HTTPException

from src.models.fem import FEMFrequencyResult, FEMMeshInfo, FEMSimulationRequest, FEMSimulationResult
from src.simulation.fem import solve_fem

router = APIRouter()


@router.post("/fem/simulate", response_model=FEMSimulationResult)
async def fem_simulate(request_body: FEMSimulationRequest) -> FEMSimulationResult:
    """Run the reduced frequency-domain FEM solver for a scripted ModelIR."""
    try:
        result = solve_fem(
            model_ir=request_body.model_ir,
            start_mhz=request_body.frequency.start_mhz,
            stop_mhz=request_body.frequency.stop_mhz,
            steps=request_body.frequency.steps,
            mesh_size_m=request_body.mesh_size_m,
            reference_impedance_ohms=request_body.reference_impedance_ohms,
            port_id=request_body.port_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return FEMSimulationResult(
        simulation_id=result.simulation_id,
        computed_in_ms=result.computed_in_ms,
        mesh=FEMMeshInfo(
            elements=result.elements,
            nodes=result.nodes,
            length_m=result.length_m,
            warnings=[],
        ),
        frequency_data=[FEMFrequencyResult.model_validate(point.__dict__) for point in result.frequency_data],
        resonant_frequency_mhz=result.resonant_frequency_mhz,
        warnings=result.warnings,
    )
