import numpy as np
import pytest

from ecgtda import hrv_features, persistence, synthetic, takens_embedding, tda_features


def test_takens_embedding_shape_and_content():
    x = np.arange(10.0)
    cloud = takens_embedding(x, delay=2, dimension=3)
    assert cloud.shape == (6, 3)
    np.testing.assert_array_equal(cloud[0], [0, 2, 4])
    np.testing.assert_array_equal(cloud[-1], [5, 7, 9])
    assert takens_embedding(x, delay=5, dimension=3).shape == (0, 3)


def test_persistence_finds_the_loop_of_a_circle():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    circle = np.column_stack([np.cos(t), np.sin(t), rng.normal(0, 0.01, t.size)])
    h1 = persistence(circle)[1]
    lifetimes = h1[:, 1] - h1[:, 0]
    assert lifetimes.max() > 0.8             # one dominant loop of radius ~1
    assert np.sort(lifetimes)[-2] < 0.3      # and nothing comparable besides it


@pytest.mark.parametrize("kind", synthetic.KINDS)
def test_generator_output_is_physiological_and_reproducible(kind):
    a = synthetic.generate(kind, n_beats=200, seed=1)
    b = synthetic.generate(kind, n_beats=200, seed=1)
    np.testing.assert_array_equal(a.rr, b.rr)
    assert len(a.rr) == 200 and len(a.pause_mask) == 200
    assert a.rr.min() > 0.3 and a.rr.max() < 12


def test_sinus_rhythm_has_no_pauses_and_av1_matches_it():
    nsr = synthetic.generate("nsr", seed=2)
    av1 = synthetic.generate("av1", seed=2)
    assert not nsr.pause_mask.any() and not av1.pause_mask.any()
    # first-degree block is invisible in the RR series: same atrial draw, same RR (up to PR jitter)
    assert abs(nsr.rr.mean() - av1.rr.mean()) < 0.01


def test_chf_profile_has_lower_variability_on_average():
    nsr = [hrv_features(synthetic.generate("nsr", seed=s).rr)["sdnn"] for s in range(30)]
    chf = [hrv_features(synthetic.generate("chf", seed=s).rr)["sdnn"] for s in range(30)]
    assert np.mean(chf) < 0.6 * np.mean(nsr)


def test_wenckebach_rr_shortens_within_a_cycle():
    """Classic Wenckebach signature: RR shrinks from beat to beat until the pause."""
    s = synthetic.generate("mobitz1", n_beats=400, seed=3)
    pauses, normal = s.rr[s.pause_mask], s.rr[~s.pause_mask]
    assert len(pauses) >= 20 and pauses.mean() > 1.3 * normal.mean()
    # runs of consecutive non-pause intervals of length >= 3 (cycles with >= 4 conducted beats)
    runs, run = [], []
    for rr, paused in zip(s.rr, s.pause_mask):
        if paused:
            runs.append(run)
            run = []
        else:
            run.append(rr)
    runs = [r for r in runs if len(r) >= 3]
    assert len(runs) >= 10
    assert np.mean([r[0] > r[-1] for r in runs]) > 0.7


def test_mobitz2_pause_is_about_twice_the_atrial_interval():
    s = synthetic.generate("mobitz2", n_beats=200, seed=4)
    pauses, normal = s.rr[s.pause_mask], s.rr[~s.pause_mask]
    assert len(pauses) > 5
    assert 1.7 < pauses.mean() / normal.mean() < 2.3


def test_pavb_pauses_are_long_and_counted():
    s = synthetic.generate("pavb", n_beats=300, seed=5, n_episodes=3)
    assert s.pause_mask.sum() == 3
    assert s.rr[s.pause_mask].min() > 2.0


def test_topological_features_react_to_a_pause():
    base = synthetic.generate("nsr", n_beats=64, seed=6).rr
    with_pause = base.copy()
    with_pause[30] = 4.0
    f0, f1 = tda_features(base), tda_features(with_pause)
    assert f1["h0_max"] > 5 * f0["h0_max"]
