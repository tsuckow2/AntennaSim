"""Reduced frequency-domain FEM solver for the M4 API contract.

This is a deterministic 1D linear-element solve along the selected lumped-port
axis. It exercises the ModelIR/material/port/frequency interfaces and returns
S11, but is not a substitute for the planned 3D H(curl) edge-element solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import uuid

import numpy as np
from scipy.linalg import solve


_SPEED_OF_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class FEMFrequencyPoint:
    frequency_mhz: float
    impedance_real_ohms: float
    impedance_imag_ohms: float
    s11_real: float
    s11_imag: float
    s11_db: float


@dataclass(frozen=True)
class FEMSolve:
    simulation_id: str
    computed_in_ms: float
    elements: int
    nodes: int
    length_m: float
    frequency_data: list[FEMFrequencyPoint]
    resonant_frequency_mhz: float | None
    warnings: list[str]


def _vec(value: Any, path: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 3 or not all(isinstance(v, (int, float)) for v in value):
        raise ValueError(f"{path} must be a finite 3-vector")
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{path} must be a finite 3-vector")
    return result


def _model_inputs(model_ir: dict[str, Any], port_id: str | None) -> tuple[float, np.ndarray, np.ndarray]:
    materials = model_ir.get("materials")
    ports = model_ir.get("ports")
    entities = model_ir.get("entities")
    if not isinstance(materials, list) or not materials:
        raise ValueError("model_ir.materials must contain at least one material")
    if not isinstance(ports, list) or not ports:
        raise ValueError("model_ir.ports must contain at least one lumped port")
    if not isinstance(entities, list) or not entities:
        raise ValueError("model_ir.entities must contain at least one geometry entity")

    port = next((item for item in ports if item.get("id") == port_id), None) if port_id else ports[0]
    if not isinstance(port, dict):
        raise ValueError(f"unknown port_id {port_id!r}")
    positive = _vec(port.get("positive"), "port.positive")
    negative = _vec(port.get("negative"), "port.negative")
    length = float(np.linalg.norm(positive - negative))
    if not math.isfinite(length) or length <= 0:
        raise ValueError("lumped-port terminals must be distinct")

    material = materials[0]
    epsilon_r = float(material.get("epsilon_r", 1.0))
    mu_r = float(material.get("mu_r", 1.0))
    if epsilon_r < 1 or mu_r <= 0:
        raise ValueError("the first material must have epsilon_r >= 1 and mu_r > 0")
    return length, positive, negative


def solve_fem(
    model_ir: dict[str, Any],
    start_mhz: float,
    stop_mhz: float,
    steps: int,
    mesh_size_m: float,
    reference_impedance_ohms: float,
    port_id: str | None = None,
) -> FEMSolve:
    """Assemble and solve a damped 1D frequency-domain FEM line model."""
    import time

    started = time.perf_counter()
    length, _, _ = _model_inputs(model_ir, port_id)
    elements = max(2, min(300, math.ceil(length / mesh_size_m)))
    nodes = elements + 1
    element_length = length / elements

    stiffness = np.zeros((nodes, nodes), dtype=complex)
    mass = np.zeros((nodes, nodes), dtype=complex)
    local_k = np.array([[1, -1], [-1, 1]], dtype=complex) / element_length
    local_m = element_length * np.array([[2, 1], [1, 2]], dtype=complex) / 6
    for index in range(elements):
        sl = np.ix_([index, index + 1], [index, index + 1])
        stiffness[sl] += local_k
        mass[sl] += local_m

    source = np.zeros(nodes, dtype=complex)
    source[0] = 1.0
    source[-1] = -1.0
    frequencies = np.linspace(start_mhz, stop_mhz, steps)
    points: list[FEMFrequencyPoint] = []
    for frequency_mhz in frequencies:
        wavenumber = 2 * math.pi * frequency_mhz * 1e6 / _SPEED_OF_LIGHT
        matrix = stiffness - (wavenumber**2) * mass
        matrix[0, 0] += 1j * 1e-6
        matrix[-1, -1] += 1j * 1e-6
        voltage = solve(matrix, source, assume_a="gen")
        impedance = complex(voltage[0] - voltage[-1])
        reflection = (impedance - reference_impedance_ohms) / (impedance + reference_impedance_ohms)
        magnitude = max(abs(reflection), 1e-12)
        points.append(FEMFrequencyPoint(
            frequency_mhz=float(frequency_mhz),
            impedance_real_ohms=float(impedance.real),
            impedance_imag_ohms=float(impedance.imag),
            s11_real=float(reflection.real),
            s11_imag=float(reflection.imag),
            s11_db=float(20 * math.log10(magnitude)),
        ))

    resonant = min(points, key=lambda point: abs(point.impedance_imag_ohms)).frequency_mhz
    return FEMSolve(
        simulation_id=uuid.uuid4().hex,
        computed_in_ms=(time.perf_counter() - started) * 1000,
        elements=elements,
        nodes=nodes,
        length_m=length,
        frequency_data=points,
        resonant_frequency_mhz=resonant,
        warnings=["fem-reduced uses a 1D port-axis linear-element model; 3D H(curl) edge elements are the next solver milestone."],
    )
