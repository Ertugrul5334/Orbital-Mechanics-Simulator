"""Run every example, regenerate all figures in docs/images and write docs/RESULTS.md.

Usage:  python run_all.py
"""

import importlib.util
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "examples"))
sys.path.insert(0, ROOT)

EXAMPLES = ["ex01_two_body", "ex02_hohmann", "ex03_plane_change", "ex04_attitude_pid",
            "ex05_j2_perturbation", "ex06_drag_decay", "ex07_mission_planning"]


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "examples", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    res = {}
    for name in EXAMPLES:
        t0 = time.time()
        res[name] = load(name).run(verbose=True)
        print(f"[{name}: {time.time() - t0:.1f} s]\n")
    write_report(res)
    print("Wrote docs/RESULTS.md")


def write_report(res):
    e1, e2, e3, e4, e5, e6, e7 = (res[n] for n in EXAMPLES)
    L = ["# Results", "",
         "Generated automatically by `python run_all.py`. All numbers come from the simulator.", ""]

    L += ["## 1. Two-body propagation (ISS-like orbit, 24 h)", "",
          "| Quantity | Value |", "|---|---|",
          f"| Propagated | {e1['duration_s']:.0f} s ({e1['orbits']:.2f} orbits, {e1['samples']} samples) |",
          f"| Element round-trip error r,v → 6 COE → r,v (worst) | {e1['roundtrip_err_m']:.1e} m |",
          f"| DOP853 position error vs analytic Kepler after 24 h | {e1['dop853_err_m']:.2e} m |",
          f"| Relative energy drift over 24 h | {e1['energy_drift_rel']:.1e} |",
          f"| Hand-written RK4: measured order of convergence | {e1['rk4_order']:.2f} (theory: 4) |",
          f"| Ground-track westward shift per orbit | {e1['gt_shift_deg']:.2f}° |", "",
          "RK4 position error after one orbit:", "", "| Step [s] | Error [m] |", "|---|---|"]
    L += [f"| {h:g} | {err:.3f} |" for h, err in e1["rk4_err_m"].items()]
    L += ["", "![3D orbit](images/01_orbit_3d.png)", "![Ground track](images/01_ground_track.png)",
          "![3D ground track](images/01_ground_track_3d.png)", "![RK4 convergence](images/01_rk4_convergence.png)", ""]

    L += ["## 2. Hohmann transfers: analytic vs numerical", "",
          "| Case | Integrator | Δv₁ [km/s] | Δv₂ [km/s] | Total [km/s] | TOF [h] | Arrival radius error | Δv₂ error |",
          "|---|---|---|---|---|---|---|---|"]
    for r in e2["rows"]:
        L.append(f"| {r['case']} | {r['method']} | {r['dv1']:+.4f} | {r['dv2']:+.4f} | {r['dv_total']:.4f} | "
                 f"{r['tof_h']:.3f} | {r['radius_err_pct']:.1e} % | {r['dv2_err_pct']:.1e} % |")
    L += ["", f"Bi-elliptic transfer (r_b = 2.5 r₂) becomes cheaper than Hohmann above r₂/r₁ ≈ "
              f"{e2['bielliptic_crossover']:.1f}.", "",
          "![Hohmann](images/02_hohmann_3d.png)", "![Bi-elliptic](images/02_hohmann_vs_bielliptic.png)", ""]

    L += ["## 3. Single-burn plane changes", "",
          "| Case | v [km/s] | Δv formula 2v·sin(Δi/2) [km/s] | Δv simulated [km/s] | Inclination after [°] |",
          "|---|---|---|---|---|"]
    for r in e3["rows"]:
        L.append(f"| {r['case']} | {r['v']:.4f} | {r['dv_analytic']:.4f} | {r['dv_sim']:.4f} | {r['i_result']:.4f} |")
    L += ["", "![Plane change](images/03_plane_change_3d.png)", "![Plane change cost](images/03_plane_change_cost.png)", ""]

    L += ["## 4. Reaction-wheel PID attitude control", "",
          "Spacecraft J = diag(2.0, 2.5, 1.8) kg·m², wheels limited to 0.2 N·m and 1 N·m·s.", "",
          "| Quantity | Value |", "|---|---|",
          f"| Slew (roll 20°, pitch −15°, yaw 30°) | {e4['slew_deg']:.1f}° eigen-axis |",
          f"| Settling time (2 % band) | **{e4['settling_s']:.2f} s** |",
          f"| Time to stay within 0.1° | {e4['settling_0p1deg_s']:.2f} s |",
          f"| Overshoot | {e4['overshoot_deg']:.3f}° |",
          f"| Peak torque / wheel momentum | {e4['peak_torque_Nm']:.2f} N·m / {e4['peak_wheel_momentum']:.3f} N·m·s |",
          f"| Steady-state error with disturbance torque, PD | {e4['pd_ss_err_deg']:.4f}° |",
          f"| Steady-state error with disturbance torque, PID | {e4['pid_ss_err_deg']:.5f}° |", "",
          "| ωn [rad/s] | Settling [s] | Overshoot [°] | Peak wheel momentum [N·m·s] |", "|---|---|---|---|"]
    L += [f"| {wn:.1f} | {ts:.2f} | {os_:.3f} | {h:.3f} |" for wn, ts, os_, h in e4["sweep"]]
    L += ["", "![Attitude](images/04_attitude_pid.png)", "![PD vs PID](images/04_pd_vs_pid.png)", ""]

    L += [f"## 5. J2 perturbation ({e5['days']}-day numerical runs vs secular theory)", "",
          f"Sun-synchronous inclination at 700 km: **{e5['sso_incl_deg']:.3f}°** "
          f"(required precession {e5['sso_required_deg_day']:.4f}°/day).", "",
          "| Orbit | i [°] | RAAN rate sim [°/day] | RAAN rate theory [°/day] | Difference |", "|---|---|---|---|---|"]
    for r in e5["rows"]:
        L.append(f"| {r['case']} | {r['i_deg']:.2f} | {r['raan_sim']:+.4f} | {r['raan_theory']:+.4f} | {r['raan_err_pct']:.2f} % |")
    L += ["", "![J2 drift](images/05_j2_raan_drift.png)", "![SSO inclination](images/05_sso_inclination.png)", ""]

    L += [f"## 6. Drag decay from {e6['start_alt']:.0f} km (J2 + drag, re-entry at 100 km)", "",
          "| Spacecraft | Cd·A/m [m²/kg] | Lifetime [days] | Orbits |", "|---|---|---|---|"]
    for r in e6["rows"]:
        L.append(f"| {r['case']} | {r['ballistic']:.4f} | {r['lifetime_days']:.1f} | {r['orbits']:.0f} |")
    L += ["", "![Drag](images/06_drag_decay.png)", ""]

    L += ["## 7. Mission planning: 300 km / 28.5° parking orbit → GEO (2,000 kg, Isp 320 s, with J2)", "",
          "| Strategy | Total Δv [m/s] | Propellant [kg] | Final a [km] | Final i [°] |", "|---|---|---|---|---|"]
    for name, r in e7["results"].items():
        import math
        L.append(f"| {name} | {r.total_dv * 1000:,.1f} | {r.total_propellant:,.1f} | "
                 f"{r.final_elements.a:,.3f} | {math.degrees(r.final_elements.i):.4f} |")
    L += ["", f"Analytic optimum: remove {e7['optimal_split_deg']:.2f}° at perigee → {e7['optimal_dv'] * 1000:,.1f} m/s.", ""]
    for name, r in e7["results"].items():
        L += [f"**{name}**", "", "```", r.summary(), "```", ""]
    L += ["![Mission](images/07_mission_3d.png)", "![Strategies](images/07_strategy_dv.png)",
          "![Split](images/07_plane_change_split.png)", ""]

    with open(os.path.join(ROOT, "docs", "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
