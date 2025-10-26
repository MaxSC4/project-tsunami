import numpy as np
import pytest

from tsunami.geo import R_EARTH, arc_length_m, great_circle_points

def test_arc_length_same_point_is_zero():
    assert arc_length_m(10.0, 20.0, 10.0, 20.0) == 0.0

def test_arc_length_equator_90deg():
    # From (0,0) to (0,90): quarter circumference = (pi/2)*R
    d = arc_length_m(0.0, 0.0, 0.0, 90.0)
    expected = (np.pi / 2.0) * R_EARTH
    assert np.isclose(d, expected, rtol=1e-7, atol=1e-4)

def test_arc_length_antipodes():
    # From (0,0) to (0,180): half circumference = pi*R
    d = arc_length_m(0.0, 0.0, 0.0, 180.0)
    expected = np.pi * R_EARTH
    assert np.isclose(d, expected, rtol=1e-7, atol=1e-3)

def test_great_circle_points_count_and_endpoints():
    lat1, lon1 = 10.0, 45.0
    lat2, lon2 = 35.0, 22.0
    n = 501
    pts = great_circle_points(lat1, lon1, lat2, lon2, npts=n)
    assert pts.shape == (n, 2)
    # endpoints (allow tiny float noise)
    assert np.isclose(pts[0, 0], lat1, rtol=0, atol=1e-10)
    assert np.isclose(pts[0, 1], lon1, rtol=0, atol=1e-10)
    assert np.isclose(pts[-1, 0], lat2, rtol=0, atol=1e-10)
    assert np.isclose(pts[-1, 1], lon2, rtol=0, atol=1e-10)

def test_great_circle_cumulative_distance_matches_arc_length():
    lat1, lon1 = -12.0, 150.0
    lat2, lon2 =  37.0, 140.0
    n = 401
    pts = great_circle_points(lat1, lon1, lat2, lon2, npts=n)
    # cumulative using your arc_length function between successive points
    seg = [
        arc_length_m(pts[k,0], pts[k,1], pts[k+1,0], pts[k+1,1])
        for k in range(n-1)
    ]
    d_sum = float(np.sum(seg))
    d_true = arc_length_m(lat1, lon1, lat2, lon2)
    # Because points are uniform in central angle, this should match very closely
    assert np.isclose(d_sum, d_true, rtol=1e-9, atol=1e-6)

@pytest.mark.xfail(reason="Known limitation: longitudes near poles may be ill-defined in current implementation.")
def test_great_circle_points_from_pole():
    # Longitude can be undefined at the pole; document current behavior.
    pts = great_circle_points(90.0, 10.0, 45.0, 21.0, npts=100)
    assert np.all(np.isfinite(pts))
