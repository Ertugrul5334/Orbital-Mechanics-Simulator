"""Example 3 - single-burn plane-change manoeuvres.

Applies an inclination change at the ascending node of a circular orbit by
rotating the velocity vector, propagates the result, and compares the burn
size and resulting inclination with the analytic 2 v sin(di/2) formula.
Also shows why plane changes are done high up, where the orbital speed is low.
"""

import _common  # noqa: F401
from _common import img

import math

import matplotlib.pyplot as plt
import numpy as np

from orbital_sim import OrbitalElements, R_EARTH, coe_to_rv, plane_change_dv, propagate, rv_to_coe
from orbital_sim import plotting
from orbital_sim.maneuvers import apply_plane_change, circular_speed

CASES = [  # name, altitude [km], i_from, i_to
    ("LEO 400 km: 51.6 -> 45 deg", 400, 51.6, 45.0),
    ("LEO 300 km: 28.5 -> 0 deg", 300, 28.5, 0.0),
    ("GEO: 28.5 -> 0 deg", 42164.0 - R_EARTH, 28.5, 0.0),
    ("LEO 700 km: 97 -> 98.2 deg", 700, 97.0, 98.2),
]


def run(verbose=True):
    rows = []
    for name, alt, i0, i1 in CASES:
        el = OrbitalElements.circular(alt, i_deg=i0, raan_deg=20.0, u_deg=0.0)   # at ascending node
        r, v = coe_to_rv(el)
        _, v_new, dv_vec = apply_plane_change(r, v, math.radians(i1))
        after = propagate(r, v_new, el.period, dt=20.0)
        el_after = rv_to_coe(*after.final_state)
        dv_analytic = plane_change_dv(circular_speed(el.a), math.radians(i1 - i0))
        rows.append({
            "case": name, "v": circular_speed(el.a), "dv_analytic": dv_analytic,
            "dv_sim": np.linalg.norm(dv_vec), "i_result": math.degrees(el_after.i),
            "i_err_deg": abs(math.degrees(el_after.i) - i1),
            "e_after": el_after.e, "a_change_km": el_after.a - el.a,
        })
        if name.startswith("LEO 400"):
            before = propagate(r, v, el.period, dt=20.0)
            ax = plotting.plot_orbits_3d([before, after], [f"i = {i0} deg", f"i = {i1} deg"],
                                         burns=[r], title="Single-burn plane change at the ascending node")
            plotting.save(ax, img("03_plane_change_3d.png"))

    # Cost of a 10 deg plane change vs altitude.
    alts = np.linspace(200, 36000, 200)
    cost = [plane_change_dv(circular_speed(R_EARTH + a), math.radians(10)) * 1000 for a in alts]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(alts, cost, color=plotting.COLORS[0])
    ax.set_xlabel("circular orbit altitude [km]"); ax.set_ylabel("dv for 10 deg plane change [m/s]")
    ax.set_title("Plane changes are cheaper where the spacecraft is slower")
    ax.grid(alpha=0.3)
    plotting.save(fig, img("03_plane_change_cost.png"))

    if verbose:
        print("=== Example 3: plane changes ===")
        print(f"{'case':<30}{'v km/s':>8}{'dv formula':>12}{'dv sim':>10}{'i after':>10}{'i err deg':>11}{'e after':>10}")
        for r in rows:
            print(f"{r['case']:<30}{r['v']:>8.4f}{r['dv_analytic']:>12.4f}{r['dv_sim']:>10.4f}"
                  f"{r['i_result']:>10.4f}{r['i_err_deg']:>11.2e}{r['e_after']:>10.2e}")
    return {"rows": rows}


if __name__ == "__main__":
    run()
