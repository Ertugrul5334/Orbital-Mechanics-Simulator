"""Example 4 - reaction-wheel PID attitude controller.

A 3-axis slew of a small satellite (J = diag(2.0, 2.5, 1.8) kg m^2, wheels
limited to 0.2 N m and 1 N m s) from rest to a new attitude, with:
  * the nominal PID response (settling time, overshoot, torque, wheel momentum),
  * PD vs PID under a constant disturbance torque (integral action removes the offset),
  * a sweep of controller bandwidth vs settling time.
"""

import _common  # noqa: F401
from _common import img

import matplotlib.pyplot as plt
import numpy as np

from orbital_sim import PIDGains, Spacecraft, quat_from_euler_deg, simulate_attitude
from orbital_sim import plotting

TARGET = (20.0, -15.0, 30.0)      # roll, pitch, yaw [deg]
DISTURBANCE = (2e-3, -2e-3, 1e-3)   # constant torque [N m] (much larger than typical, for clarity)


def run(verbose=True):
    sc = Spacecraft()
    q0 = np.array([1.0, 0.0, 0.0, 0.0])
    qt = quat_from_euler_deg(*TARGET)

    nominal = simulate_attitude(q0, qt, spacecraft=sc, duration=30.0)
    plotting.save(plotting.plot_attitude(nominal, "PID slew: roll 20, pitch -15, yaw 30 deg"),
                  img("04_attitude_pid.png"))

    gains = PIDGains.from_bandwidth(sc.inertia)
    pd = PIDGains(gains.kp, gains.ki * 0, gains.kd)
    with_pd = simulate_attitude(q0, qt, spacecraft=sc, gains=pd, disturbance=DISTURBANCE, duration=60.0)
    with_pid = simulate_attitude(q0, qt, spacecraft=sc, gains=gains, disturbance=DISTURBANCE, duration=60.0)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(with_pd.t, with_pd.error_deg, color=plotting.COLORS[1], label="PD")
    ax.semilogy(with_pid.t, with_pid.error_deg, color=plotting.COLORS[0], label="PID")
    ax.set_xlabel("time [s]"); ax.set_ylabel("pointing error [deg]")
    ax.set_title("Constant disturbance torque: integral term removes the offset")
    ax.grid(alpha=0.3, which="both"); ax.legend()
    plotting.save(fig, img("04_pd_vs_pid.png"))

    sweep = []
    for wn in (0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0):
        res = simulate_attitude(q0, qt, spacecraft=sc, gains=PIDGains.from_bandwidth(sc.inertia, wn=wn),
                                duration=60.0)
        sweep.append((wn, res.settling_time(), res.overshoot_deg(),
                      float(np.max(np.abs(res.h_wheel)))))

    out = {
        "slew_deg": float(nominal.error_deg[0]),
        "settling_s": nominal.settling_time(),
        "settling_0p1deg_s": nominal.settling_time(band_deg=0.1),
        "overshoot_deg": nominal.overshoot_deg(),
        "peak_torque_Nm": float(np.max(np.abs(nominal.torque))),
        "peak_wheel_momentum": float(np.max(np.abs(nominal.h_wheel))),
        "pd_ss_err_deg": with_pd.steady_state_error_deg(),
        "pid_ss_err_deg": with_pid.steady_state_error_deg(),
        "sweep": sweep,
    }
    if verbose:
        print("=== Example 4: reaction-wheel PID attitude control ===")
        print(f"Slew angle:                  {out['slew_deg']:.2f} deg")
        print(f"Settling time (2% band):     {out['settling_s']:.2f} s")
        print(f"Settling time (0.1 deg):     {out['settling_0p1deg_s']:.2f} s")
        print(f"Overshoot:                   {out['overshoot_deg']:.3f} deg")
        print(f"Peak torque / wheel momentum:{out['peak_torque_Nm']:.3f} N m / {out['peak_wheel_momentum']:.3f} N m s")
        print(f"Steady-state error with disturbance: PD {out['pd_ss_err_deg']:.4f} deg, "
              f"PID {out['pid_ss_err_deg']:.5f} deg")
        print("Bandwidth sweep (wn rad/s, settling s, overshoot deg, peak wheel momentum N m s):")
        for row in sweep:
            print("   " + ", ".join(f"{x:.3f}" for x in row))
    return out


if __name__ == "__main__":
    run()
