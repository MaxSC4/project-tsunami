import numpy as np
import pytest

from tsunami.speed_model import speed

G = 9.81

def test_speed_scalar_positive_depth():
    h = 4000.0
    v = speed(h)
    expected = np.sqrt(G * h)
    assert np.isclose(v, expected)

def test_speed_zero_depth():
    assert speed(0.0) == 0.0

def test_speed_vector_input():
    h = np.array([0.0, 100.0, 4000.0])
    v = speed(h)
    expected = np.sqrt(G * h)
    assert np.allclose(v, expected)

def test_speed_monotonic_for_positive_depths():
    h1, h2 = 100.0, 4000.0
    assert speed(h1) < speed(h2)

def test_speed_nan_propagation():
    v = speed(np.array([1000.0, np.nan, 2000.0]))
    assert np.isnan(v[1]) and np.isfinite(v[0]) and np.isfinite(v[2])

def test_speed_infinite_depth():
    # +inf depth -> +inf speed
    assert np.isinf(speed(np.inf))
    # -inf depth -> +inf as well, given current abs() definition
    assert np.isinf(speed(-np.inf))

def test_speed_negative_depth_symmetry_current_behavior():
    # Current implementation uses abs(-g*h), so v(h) == v(-h)
    h = 1234.5
    assert np.isclose(speed(h), speed(-h))

@pytest.mark.xfail(reason="Physical expectation: negative (land) depths should not produce a real wave speed.")
def test_speed_negative_depth_should_be_nan_or_raise():
    # Marked xfail to document the intended physical behavior if you change the function later.
    v = speed(-100.0)
    assert np.isnan(v)
