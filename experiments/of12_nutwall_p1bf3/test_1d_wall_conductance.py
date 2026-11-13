#!/usr/bin/env python3
"""Algebra-only verification of the Option-A unresolved wall layer contract.

This is NOT a tetrahedral kernel test. It checks dimensions, signs, clipping,
the viscous limit and correct wall-sample mapping for a P1 1D affine interval.
"""
from __future__ import annotations
import math


def wall_conductance(*, nu: float, y_sample: float, h: float,
                     velocity_at_sample: float, u_tau: float,
                     face_area: float = 1.0) -> tuple[float, float, float]:
    assert all(math.isfinite(x) for x in
               (nu, y_sample, h, velocity_at_sample, u_tau, face_area))
    assert nu > 0 and h > 0 and 0 < y_sample <= h and face_area > 0
    assert velocity_at_sample >= 0 and u_tau >= 0
    # When sample is at y_s = t*h, U_s = t * U_first for a linear P1 field.
    # Physical target traction (per density) is u_tau**2.
    u_first = velocity_at_sample * h / y_sample
    k_base = face_area * nu / h
    if velocity_at_sample <= 1e-14:
        # Viscous limiting value, not an artificial large turbulent viscosity.
        # A zero-speed sample with nonzero u_tau is inconsistent and rejected.
        assert u_tau <= 1e-12, "nonzero wall shear with stagnant sample"
        k_model = k_base
    else:
        nut_wall = max(0.0, u_tau*u_tau*y_sample/velocity_at_sample - nu)
        # Model resolved first-node stiffness, including the molecular part.
        k_model = face_area * (nu + nut_wall) / h
    delta = k_model - k_base
    return k_base, k_model, delta


def test_viscous_limit() -> None:
    nu = 1.460734693878e-5
    for h in (0.0001, 0.001, 0.01):
        for t in (0.25, 0.5, 1.0):
            y = h * t
            velocity = 10.0 * t
            u_tau = math.sqrt(nu * velocity / y)
            base, model, correction = wall_conductance(
                nu=nu, y_sample=y, h=h,
                velocity_at_sample=velocity, u_tau=u_tau)
            assert abs(correction) < 1e-12 * base, (base, model, correction)


def test_log_wall() -> None:
    nu, h, t = 1.460734693878e-5, 0.005, 0.4
    velocity = 7.0 * t
    utau = 0.5
    _, model, correction = wall_conductance(
        nu=nu, y_sample=h*t, h=h,
        velocity_at_sample=velocity, u_tau=utau)
    u_first = velocity/t
    # Face traction per density is model conductance times resolved speed.
    assert abs(model*u_first-utau*utau) < 1e-12
    assert correction > 0


def test_clipping_and_stagnation() -> None:
    nu, h = 1e-3, 0.01
    base, model, correction = wall_conductance(
        nu=nu, y_sample=h, h=h, velocity_at_sample=1.0, u_tau=0)
    assert base == model and correction == 0
    base, model, correction = wall_conductance(
        nu=nu, y_sample=h, h=h, velocity_at_sample=0.0, u_tau=0)
    assert base == model and correction == 0


if __name__ == "__main__":
    test_viscous_limit()
    test_log_wall()
    test_clipping_and_stagnation()
    print("OPTION_A_1D_CONDUCTANCE status=PASS; native 3D operator NOT implemented")
