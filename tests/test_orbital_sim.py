"""Validation tests against textbook values and analytic solutions.

Run with:  python -m pytest -q
"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbital_sim import (AtmosphericDrag, Coast, CoastToNode, Hohmann, J2Perturbation, Mission,
                         OrbitalElements, PlaneChange, R_EARTH, bi_elliptic, coe_to_rv,
                         combined_plane_change_dv, hohmann, j2_secular_rates, kepler_propagate,
                         plane_change_dv, propagate, quat_from_euler_deg, rv_to_coe,
                         simulate_attitude, sun_synchronous_inclination)
from orbital_sim.constants import DAY
from orbital_sim.elements import mean_to_true, true_to_mean
from orbital_sim.groundtrack import ground_track
from orbital_sim.maneuvers import apply_plane_change, apply_prograde_burn, circular_speed
from orbital_sim.perturbations import atmospheric_density


# --- Orbital elements --------------------------------------------------------
@pytest.mark.parametrize("a,e,i,raan,argp,nu", [
    (7000, 0.05, 51.6, 40, 30, 10),
    (26600, 0.74, 63.4, 300, 270, 200),
    (42164, 0.0001, 0.05, 10, 20, 30),
    (8000, 0.2, 120, 359, 1, 359),
])
def test_coe_roundtrip(a, e, i, raan, argp, nu):
    el = OrbitalElements.from_degrees(a, e, i, raan, argp, nu)
    back = rv_to_coe(*coe_to_rv(el))
    assert back.a == pytest.approx(a, rel=1e-10)
    assert back.e == pytest.approx(e, abs=1e-10)
    for x, y in [(back.i, el.i), (back.raan, el.raan), (back.argp, el.argp), (back.nu, el.nu)]:
        assert math.remainder(x - y, 2 * math.pi) == pytest.approx(0, abs=1e-8)


def test_circular_equatorial_roundtrip_state():
    el = OrbitalElements.circular(500, i_deg=0, u_deg=123)
    r, v = coe_to_rv(el)
    r2, v2 = coe_to_rv(rv_to_coe(r, v))
    assert np.allclose(r, r2, atol=1e-6) and np.allclose(v, v2, atol=1e-9)


def test_vallado_example_2_5():
    # Vallado Example 2-5: r = [6524.834, 6862.875, 6448.296] km, v = [4.901327, 5.533756, -1.976341] km/s
    el = rv_to_coe([6524.834, 6862.875, 6448.296], [4.901327, 5.533756, -1.976341])
    assert el.p == pytest.approx(11067.790, abs=0.01)
    assert el.a == pytest.approx(36127.343, abs=0.01)
    assert el.e == pytest.approx(0.832853, abs=1e-6)
    assert math.degrees(el.i) == pytest.approx(87.870, abs=1e-3)
    assert math.degrees(el.raan) == pytest.approx(227.89, abs=1e-2)
    assert math.degrees(el.argp) == pytest.approx(53.38, abs=1e-2)
    assert math.degrees(el.nu) == pytest.approx(92.335, abs=1e-3)


def test_kepler_equation_inverse():
    for e in (0.0, 0.1, 0.5, 0.9, 0.99):
        for nu in np.linspace(0, 2 * np.pi, 13)[:-1]:
            assert math.remainder(mean_to_true(true_to_mean(nu, e), e) - nu, 2 * math.pi) == \
                pytest.approx(0, abs=1e-9)


# --- Propagation ---------------------------------------------------------------
def test_propagation_matches_kepler_and_conserves_energy():
    el = OrbitalElements.from_degrees(7500, 0.1, 45, 10, 20, 0)
    r0, v0 = coe_to_rv(el)
    traj = propagate(r0, v0, 3 * el.period, dt=30)
    r_exact = coe_to_rv(kepler_propagate(el, traj.t[-1]))[0]
    assert np.linalg.norm(traj.r[-1] - r_exact) < 1e-3          # < 1 m after 3 orbits
    E = traj.energy()
    assert np.ptp(E) / abs(E[0]) < 1e-9
    H = traj.angular_momentum()
    assert np.ptp(H) / H[0] < 1e-9


def test_rk4_is_fourth_order():
    el = OrbitalElements.circular(400)
    r0, v0 = coe_to_rv(el)
    errs = []
    for h in (60, 30):
        tr = propagate(r0, v0, el.period, dt=h, method="rk4")
        errs.append(np.linalg.norm(tr.r[-1] - coe_to_rv(kepler_propagate(el, tr.t[-1]))[0]))
    assert 12 < errs[0] / errs[1] < 24       # halving step -> ~16x smaller error


def test_propagates_over_1000_seconds():
    r0, v0 = coe_to_rv(OrbitalElements.circular(400))
    tr = propagate(r0, v0, 1000.0, dt=1.0)
    assert tr.t[-1] == pytest.approx(1000.0) and len(tr.t) == 1001


# --- Manoeuvres ----------------------------------------------------------------
def test_hohmann_leo_geo_textbook():
    # Vallado Example 6-1: 191.34411 km altitude -> 35781.34857 km altitude.
    h = hohmann(6569.4781, 42159.4856)
    assert h.dv1 == pytest.approx(2.457038, abs=1e-5)
    assert h.dv2 == pytest.approx(1.478187, abs=1e-5)
    assert h.tof / 60 == pytest.approx(315.4, abs=0.5)


def test_hohmann_numerical_matches_analytic():
    r1, r2 = R_EARTH + 300, 42164.0
    plan = hohmann(r1, r2)
    r0, v0 = coe_to_rv(OrbitalElements(r1, 0, 0, 0, 0, 0))
    _, v_after, _ = apply_prograde_burn(r0, v0, plan.dv1)
    arc = propagate(r0, v_after, plan.tof, dt=10)
    r_arr, v_arr = arc.final_state
    assert abs(np.linalg.norm(r_arr) - r2) / r2 < 1e-6
    assert (circular_speed(r2) - np.linalg.norm(v_arr)) == pytest.approx(plan.dv2, rel=1e-6)


def test_bi_elliptic_beats_hohmann_for_large_ratio():
    r1 = 7000.0
    assert bi_elliptic(r1, 20 * r1, 60 * r1).dv_total < hohmann(r1, 20 * r1).dv_total
    assert bi_elliptic(r1, 5 * r1, 15 * r1).dv_total > hohmann(r1, 5 * r1).dv_total


def test_plane_change():
    el = OrbitalElements.circular(400, i_deg=51.6, raan_deg=33, u_deg=0)
    r, v = coe_to_rv(el)
    _, v_new, dv = apply_plane_change(r, v, math.radians(45.0))
    assert np.linalg.norm(dv) == pytest.approx(plane_change_dv(np.linalg.norm(v), math.radians(6.6)), rel=1e-9)
    after = rv_to_coe(r, v_new)
    assert math.degrees(after.i) == pytest.approx(45.0, abs=1e-9)
    assert math.degrees(after.raan) == pytest.approx(33.0, abs=1e-9)
    assert combined_plane_change_dv(3, 3, math.radians(10)) == pytest.approx(plane_change_dv(3, math.radians(10)))


def test_mission_targeting_with_j2_hits_geo():
    m = Mission(OrbitalElements.circular(300, i_deg=28.5, u_deg=45), perturbations=[J2Perturbation()])
    m.add(CoastToNode()).add(Hohmann(42164.0, final_inclination_deg=0.0)).add(PlaneChange(0.0))
    res = m.run()
    assert res.final_elements.a == pytest.approx(42164.0, abs=1.0)
    assert res.final_elements.e < 1e-3
    assert math.degrees(res.final_elements.i) < 1e-6
    assert 4.20 < res.total_dv < 4.30
    assert res.total_propellant < 1000


# --- Perturbations ------------------------------------------------------------
def test_sun_synchronous_inclination():
    assert math.degrees(sun_synchronous_inclination(R_EARTH + 700)) == pytest.approx(98.19, abs=0.02)
    assert math.degrees(sun_synchronous_inclination(R_EARTH + 800)) == pytest.approx(98.60, abs=0.02)


def test_j2_raan_drift_matches_theory():
    a = R_EARTH + 700
    el = OrbitalElements(a, 0.0, sun_synchronous_inclination(a), 0, 0, 0)
    tr = propagate(*coe_to_rv(el), 5 * DAY, dt=600, perturbations=[J2Perturbation()], rtol=1e-10, atol=1e-10)
    hist = tr.element_history()
    rate = np.polyfit(hist["t"] / DAY, hist["raan"], 1)[0]
    assert rate == pytest.approx(0.9856, rel=0.01)            # deg/day, within 1 %
    theory = math.degrees(j2_secular_rates(a, 0.0, el.i)[0]) * DAY
    assert theory == pytest.approx(0.9856, abs=1e-4)


def test_j2_keeps_energy_conservative():
    el = OrbitalElements.circular(500, i_deg=60)
    tr = propagate(*coe_to_rv(el), DAY, dt=60, perturbations=[J2Perturbation()])
    r = tr.radius
    # Energy including the J2 potential is conserved in an inertial frame.
    from orbital_sim.constants import J2_EARTH, MU_EARTH
    sin_phi = tr.r[:, 2] / r
    U = -MU_EARTH / r * (1 - J2_EARTH * (R_EARTH / r) ** 2 * (1.5 * sin_phi ** 2 - 0.5))
    E = 0.5 * tr.speed ** 2 + U
    assert np.ptp(E) / abs(E[0]) < 1e-9


def test_drag_lowers_orbit_and_density_model():
    assert atmospheric_density(0) == pytest.approx(1.225)
    assert atmospheric_density(400) == pytest.approx(3.725e-12)
    assert atmospheric_density(450) < atmospheric_density(400)
    el = OrbitalElements.circular(250, i_deg=51.6)
    tr = propagate(*coe_to_rv(el), DAY, dt=60, perturbations=[AtmosphericDrag(2.2, 0.03, 4.0)])
    a_end = rv_to_coe(*tr.final_state).a
    assert a_end < el.a - 1.0


# --- Ground track --------------------------------------------------------------
def test_ground_track_shift_per_orbit():
    el = OrbitalElements.circular(400, i_deg=51.6)
    tr = propagate(*coe_to_rv(el), el.period, dt=10)
    lat, lon, _ = ground_track(tr.r, tr.t)
    assert abs(lat).max() == pytest.approx(51.6, abs=0.1)
    shift = (lon[0] - lon[-1]) % 360
    assert shift == pytest.approx(math.degrees(7.2921159e-5 * el.period), abs=0.1)


# --- Attitude control --------------------------------------------------------------
def test_pid_settles_within_5_to_10_seconds():
    res = simulate_attitude([1, 0, 0, 0], quat_from_euler_deg(20, -15, 30), duration=30)
    ts = res.settling_time()
    assert 3.0 < ts < 10.0
    assert res.error_deg[-1] < 0.05
    assert np.max(np.abs(res.torque)) <= 0.2 + 1e-12


def test_integral_action_removes_disturbance_offset():
    from orbital_sim import PIDGains, Spacecraft
    sc = Spacecraft()
    g = PIDGains.from_bandwidth(sc.inertia)
    pd = PIDGains(g.kp, g.ki * 0, g.kd)
    kw = dict(spacecraft=sc, disturbance=(2e-3, -2e-3, 1e-3), duration=60)
    q0, qt = [1, 0, 0, 0], quat_from_euler_deg(10, 0, 0)
    e_pd = simulate_attitude(q0, qt, gains=pd, **kw).steady_state_error_deg()
    e_pid = simulate_attitude(q0, qt, gains=g, **kw).steady_state_error_deg()
    assert e_pid < e_pd / 20
