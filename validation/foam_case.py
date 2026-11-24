"""Write and run an OpenFOAM (v11) incompressible RANS case on an AeroMesh mesh.

The solver is the referee the project did not have: ``checkMesh`` measures the
mesh with someone else's code, and the forces are compared against published
values rather than against our own metrics.

Spalart-Allmaras, steady, wall-resolved (nut = 0 at the wall, no wall
function), freestream nuTilda = 3 nu as on the NASA Turbulence Modeling
Resource.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

from aeromesh.mesh.export import write_openfoam

FOAM = "/opt/openfoam11/etc/bashrc"

_HEAD = """FoamFile
{{
    format      ascii;
    class       {cls};
    object      {obj};
}}
"""


def _field(cls, obj, dims, internal, patches):
    body = _HEAD.format(cls=cls, obj=obj)
    body += f"dimensions      {dims};\ninternalField   {internal};\n"
    body += "boundaryField\n{\n"
    for name, text in patches.items():
        body += f"    {name}\n    {{\n{text}    }}\n"
    return body + "}\n"


def write_case(mesh, case, reynolds=6e6, alpha_deg=0.0, n_iter=6000,
               write_interval=None):
    case = Path(case)
    (case / "0").mkdir(parents=True, exist_ok=True)
    (case / "system").mkdir(exist_ok=True)
    patches = write_openfoam(mesh, case)
    nu = 1.0 / reynolds
    a = np.deg2rad(alpha_deg)
    u = f"({np.cos(a):.12g} {np.sin(a):.12g} 0)"
    free = [p for p in patches if p not in ("wall", "frontAndBack")]
    empty = "        type            empty;\n"

    def bc(wall, far):
        d = {p: far for p in free}
        d["wall"] = wall
        d["frontAndBack"] = empty
        return d

    (case / "0" / "U").write_text(_field(
        "volVectorField", "U", "[0 1 -1 0 0 0 0]", f"uniform {u}",
        bc("        type            noSlip;\n",
           "        type            freestreamVelocity;\n"
           f"        freestreamValue uniform {u};\n")))
    (case / "0" / "p").write_text(_field(
        "volScalarField", "p", "[0 2 -2 0 0 0 0]", "uniform 0",
        bc("        type            zeroGradient;\n",
           "        type            freestreamPressure;\n"
           "        freestreamValue uniform 0;\n")))
    nt = 3.0 * nu
    (case / "0" / "nuTilda").write_text(_field(
        "volScalarField", "nuTilda", "[0 2 -1 0 0 0 0]", f"uniform {nt:.8g}",
        bc("        type            fixedValue;\n        value uniform 0;\n",
           "        type            freestream;\n"
           f"        freestreamValue uniform {nt:.8g};\n")))
    nut = nt * 27.0 / (27.0 + 7.1 ** 3)
    (case / "0" / "nut").write_text(_field(
        "volScalarField", "nut", "[0 2 -1 0 0 0 0]", f"uniform {nut:.8g}",
        bc("        type            fixedValue;\n        value uniform 0;\n",
           "        type            freestream;\n"
           f"        freestreamValue uniform {nut:.8g};\n")))

    (case / "constant" / "momentumTransport").write_text(
        _HEAD.format(cls="dictionary", obj="momentumTransport")
        + "simulationType RAS;\nRAS\n{\n    model SpalartAllmaras;\n"
          "    turbulence on;\n    printCoeffs off;\n}\n")
    (case / "constant" / "physicalProperties").write_text(
        _HEAD.format(cls="dictionary", obj="physicalProperties")
        + f"viscosityModel constant;\nnu [0 2 -1 0 0 0 0] {nu:.10g};\n")

    wi = write_interval or n_iter
    (case / "system" / "controlDict").write_text(
        _HEAD.format(cls="dictionary", obj="controlDict") + f"""
application     foamRun;
solver          incompressibleFluid;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         {n_iter};
deltaT          1;
writeControl    timeStep;
writeInterval   {wi};
purgeWrite      1;
writeFormat     ascii;
writePrecision  8;
timeFormat      general;
runTimeModifiable true;
functions
{{
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         (wall);
        rho             rhoInf;
        rhoInf          1;
        CofR            (0.25 0 0);
        liftDir         ({-np.sin(a):.12g} {np.cos(a):.12g} 0);
        dragDir         ({np.cos(a):.12g} {np.sin(a):.12g} 0);
        pitchAxis       (0 0 1);
        magUInf         1;
        lRef            1;
        Aref            1;
    }}
    yPlus
    {{
        type            yPlus;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
    }}
    wallShearStress
    {{
        type            wallShearStress;
        libs            ("libfieldFunctionObjects.so");
        writeControl    writeTime;
        patches         (wall);
    }}
}}
""")
    (case / "system" / "fvSchemes").write_text(
        _HEAD.format(cls="dictionary", obj="fvSchemes") + """
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; grad(U) cellLimited Gauss linear 1;
                  grad(nuTilda) cellLimited Gauss linear 1; }
divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,nuTilda) bounded Gauss linearUpwind grad(nuTilda);
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }
wallDist        { method meshWave; }
""")
    (case / "system" / "fvSolution").write_text(
        _HEAD.format(cls="dictionary", obj="fvSolution") + """
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-9;
        relTol          0.05;
        maxIter         100;
    }
    "(U|nuTilda)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        nSweeps         2;
        tolerance       1e-10;
        relTol          0.1;
    }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    residualControl { p 1e-7; U 1e-8; nuTilda 1e-8; }
}
relaxationFactors
{
    equations { U 0.9; nuTilda 0.9; }
}
""")
    return patches


def foam(cmd: str, case, log: str | None = None, background=False):
    case = Path(case)
    full = f"source {FOAM} && cd '{case}' && {cmd}"
    if log:
        full += f" > {log} 2>&1"
    if background:
        return subprocess.Popen(["bash", "-c", full])
    return subprocess.run(["bash", "-c", full], capture_output=True, text=True)


def check_mesh(case) -> dict:
    out = foam("checkMesh", case).stdout
    Path(case, "log.checkMesh").write_text(out)

    def grab(pattern, cast=float):
        m = re.search(pattern, out)
        return cast(m.group(1)) if m else None

    return {
        "cells": grab(r"cells:\s+(\d+)", int),
        "max_aspect": grab(r"Max aspect ratio = ([\d.e+-]+)"),
        "nonorth_max": grab(r"Mesh non-orthogonality Max: ([\d.e+-]+)"),
        "nonorth_avg": grab(r"average: ([\d.e+-]+)"),
        "max_skewness": grab(r"Max skewness = ([\d.e+-]+)"),
        "ok": "Mesh OK." in out,
        "failed_checks": grab(r"Failed (\d+) mesh checks", int) or 0,
    }


def read_coeffs(case) -> np.ndarray:
    """(n, columns) history: time, Cm, Cd, Cl, ..."""
    files = sorted(Path(case, "postProcessing", "forceCoeffs").glob(
        "*/forceCoeffs*.dat"))
    rows = []
    cols = None
    for f in files:
        for line in f.read_text().splitlines():
            if line.startswith("#"):
                if "Time" in line:
                    cols = line.lstrip("# ").split()
                continue
            rows.append([float(v) for v in line.split()])
    return cols, np.array(rows)
