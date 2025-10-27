import numpy as np
import pytest

from tsunami.speed_integrator import travel_time_seconds
from tsunami.geo import arc_length_m

def fake_depth(lat, lon):
    """Fake constant ocean depth (always 4000 m)."""
    return 4000.0

def test_travel_time_constant_depth_matches_theory():
    lat1, lon1 = 0.0, 150.0
    lat2, lon2 = 35.0, 140.0

    d = arc_length_m(lat1, lon1, lat2, lon2)
    v = np.sqrt(9.81 * 4000.0)
    T_theo = d / v
    T_num = travel_time_seconds(lat1, lon1, lat2, lon2, fake_depth, n_samples=1201)

    rel_err = abs(T_num - T_theo) / T_theo
    assert rel_err < 0.01, f"Relative error too high ({rel_err*100:.3f}%)"

def test_zero_distance_returns_zero():
    T = travel_time_seconds(10, 20, 10, 20, fake_depth)
    assert T == 0.0

def test_invalid_depth_returns_inf():
    def bad_depth(lat, lon):
        return np.nan if lat > 0 else 4000.0
    T = travel_time_seconds(0, 0, 10, 10, bad_depth)
    assert np.isinf(T)

def test_travel_time_increases_with_distance():
    T1 = travel_time_seconds(0, 0, 5, 0, fake_depth)
    T2 = travel_time_seconds(0, 0, 10, 0, fake_depth)
    assert T2 > T1

def test_vectorization_consistency():
    # depth function that returns depth depending on latitude
    def varying_depth(lat, lon):
        return 4000.0 + 100.0 * np.sin(np.deg2rad(lat))
    T = travel_time_seconds(0, 150, 30, 150, varying_depth)
    assert np.isfinite(T)
