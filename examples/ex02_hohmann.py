"""Example 2 - two-burn Hohmann transfers, checked by numerical simulation.

For each test case the analytic Hohmann solution (dv1, dv2, half-period TOF)
is compared with a full numerical run: burn 1 is applied to the state vector,
the transfer arc is integrated, and the arrival radius and the burn needed to
circularise are measured from the simulated state.
"""

import _common  # noqa: F401
from _common import img

import matplotlib.pyplot as plt
import numpy as np

from orbital_sim import OrbitalElements, R_EARTH, bi_elliptic, coe_to_rv, hohmann, propagate
from orbital_sim import plotting
from orbital_sim.maneuvers import apply_prograde_burn, circular_speed, vis_viva

CASES = [
    ("LEO 300 km -> GEO", R_EARTH + 300, 42164.0),
    ("LEO 400 km -> 800 km", R_EARTH + 400, R_EARTH + 800),
    ("LEO 500 km -> MEO 20,200 km (GPS)", R_EARTH + 500, R_EARTH + 20200),
    ("GEO -> LEO 400 km (lowering)", 42164.0, R_EARTH + 400),
]


def simulate_hohmann(r1, r2, method="dop853", dt=10.0):
    plan = hohmann(r1, r2)
    r0, v0 = coe_to_rv(OrbitalElements(r1, 0.0, 0.0, 0.0, 0.0, 0.0))
    _, v_after, _ = apply_prograde_burn(r0, v0, plan.dv1)
    arc = propagate(r0, v_after, plan.tof, dt=dt, method=method, label="transfer")
    r_arr, v_arr = arc.final_state
    r_arr_n = np.linalg.norm(r_arr)
    dv2_num = circular_speed(r_arr_n) - np.linalg.norm(v_arr)
    initial = propagate(r0, v0, 2 * np.pi * np.sqrt(r1 ** 3 / 398600.4418), dt=30, label="initial")
    v_final = v_arr / np.linalg.norm(v_arr) * circular_speed(r_arr_n)
    final = propagate(r_arr, v_final, 2 * np.pi * np.sqrt(r_arr_n ** 3 / 398600.4418), dt=30, label="final")
    return plan, arc, initial, final, r_arr_n, dv2_num


def run(verbose=True):
    rows = []
    for name, r1, r2 in CASES:
        for method, dt in (("dop853", 10.0), ("rk4", 60.0)):
            plan, arc, initial, final, r_arr, dv2_num = simulate_hohmann(r1, r2, method, dt)
            rows.append({
                "case": name, "method": f"{method.upper()} (dt={dt:g}s)" if method == "rk4" else "DOP853",
                "dv1": plan.dv1, "dv2": plan.dv2, "dv_total": plan.dv_total, "tof_h": plan.tof / 3600,
                "r_arrival": r_arr, "radius_err_pct": 100 * abs(r_arr - r2) / r2,
                "dv2_num": dv2_num, "dv2_err_pct": 100 * abs(dv2_num - plan.dv2) / abs(plan.dv2),
            })
            if name == CASES[0][0] and method == "dop853":
                burns = [initial.r[0], arc.r[-1]]
                ax = plotting.plot_orbits_3d([initial, arc, final],
                                             ["LEO 300 km", "transfer ellipse", "GEO"],
                                             burns=burns, title="Hohmann transfer LEO -> GEO")
                ax.view_init(elev=60, azim=-60)
                plotting.save(ax, img("02_hohmann_3d.png"))

    # Hohmann vs bi-elliptic across radius ratios (crossover ~11.94).
    r1 = R_EARTH + 300
    ratios = np.linspace(2, 30, 120)
    hoh = np.array([hohmann(r1, q * r1).dv_total for q in ratios]) / circular_speed(r1)
    bie = np.array([bi_elliptic(r1, q * r1, 2.5 * q * r1).dv_total for q in ratios]) / circular_speed(r1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ratios, hoh, color=plotting.COLORS[0], label="Hohmann")
    ax.plot(ratios, bie, color=plotting.COLORS[1], label="Bi-elliptic (rb = 2.5 r2)")
    ax.set_xlabel("r2 / r1"); ax.set_ylabel("total dv / v_circ(r1)")
    ax.set_title("Hohmann vs bi-elliptic transfer cost"); ax.grid(alpha=0.3); ax.legend()
    plotting.save(fig, img("02_hohmann_vs_bielliptic.png"))
    cross = ratios[np.argmax(bie < hoh)] if np.any(bie < hoh) else None

    if verbose:
        print("=== Example 2: Hohmann transfers (analytic vs numerical) ===")
        hdr = f"{'case':<36}{'integrator':<18}{'dv1':>8}{'dv2':>8}{'total':>8}{'TOF h':>8}{'r err %':>10}{'dv2 err %':>11}"
        print(hdr)
        for r in rows:
            print(f"{r['case']:<36}{r['method']:<18}{r['dv1']:>8.4f}{r['dv2']:>8.4f}{r['dv_total']:>8.4f}"
                  f"{r['tof_h']:>8.3f}{r['radius_err_pct']:>10.2e}{r['dv2_err_pct']:>11.2e}")
        print(f"Bi-elliptic (rb = 2.5 r2) beats Hohmann above r2/r1 ~ {cross:.1f}")
    return {"rows": rows, "bielliptic_crossover": cross}


if __name__ == "__main__":
    run()
