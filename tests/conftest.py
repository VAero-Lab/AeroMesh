"""Shared test fixtures for AeroMesh."""

from pathlib import Path

import numpy as np
import pytest

from aeromesh.geometry.boundary import Boundary, load_airfoil

# Path to the data directory
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "airfoils"


@pytest.fixture
def naca0012() -> Boundary:
    """NACA 0012 boundary, unit chord, closed, CCW."""
    return load_airfoil("0012", num_points=100)


@pytest.fixture
def sd7037() -> Boundary:
    """SD7037 boundary from .dat file, unit chord, closed, CCW."""
    dat_path = DATA_DIR / "sd7037.dat"
    if not dat_path.exists():
        pytest.skip(f"SD7037 data file not found at {dat_path}")
    return load_airfoil(str(dat_path))
