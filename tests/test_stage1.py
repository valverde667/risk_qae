import numpy as np

from risk_qae.config import BoundsConfig, DiscretizationConfig
from risk_qae.discretization import HistogramDiscretizer
from risk_qae.metrics import classical_mean, classical_var, classical_tvar


def test_discretizer_pmf_sums_to_one():
    rng = np.random.default_rng(0)
    samples = rng.uniform(0.0, 1.0, size=20_000)

    dist = HistogramDiscretizer().fit(
        {"samples": samples},
        dcfg=DiscretizationConfig(n_index_qubits=8),
        bcfg=BoundsConfig(clip_percentiles=(0.1, 99.9)),
    )
    assert abs(float(np.sum(dist.pmf)) - 1.0) < 1e-12
    assert dist.pmf.size == 2 ** 8


def test_classical_metrics_reasonable_uniform():
    rng = np.random.default_rng(1)
    samples = rng.uniform(0.0, 1.0, size=50_000)

    dist = HistogramDiscretizer().fit({"samples": samples}, DiscretizationConfig(8), BoundsConfig((0.1, 99.9)))

    m = classical_mean(dist).mean
    assert abs(m - 0.5) < 0.05

    v = classical_var(dist, 0.95).var
    assert 0.85 <= v <= 1.0

    t = classical_tvar(dist, 0.95).tvar
    # Expected tail mean for uniform [0,1] above 0.95 is about 0.975
    assert 0.90 <= t <= 1.0
