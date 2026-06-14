"""R18 further-enhanced Type II PS predicted-PMI at N4=1."""

import numpy as np
import pytest

from nr_csi.channel import RandomRayChannel
from nr_csi.codebooks import R17Type2Codebook, R18PredictedPortSelectionCodebook
from nr_csi.codebooks.serialize import pack, unpack
from nr_csi.config import AntennaConfig
from nr_csi.eval import evaluate

ANT = AntennaConfig.standard(4, 2)


def random_channel(seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(1, 8, 4, ANT.P)) + 1j * rng.normal(size=(1, 8, 4, ANT.P))


@pytest.mark.parametrize("rank", [1, 2, 4])
def test_equivalent_to_r17(rank):
    H = random_channel(rank)
    r17 = R17Type2Codebook(ANT, N3=8, param_combination=6, N_window=4)
    predicted = R18PredictedPortSelectionCodebook(
        ANT, N3=8, param_combination=6, N_window=4
    )
    pmi17 = r17.select(H, rank)
    pmi18 = predicted.select(H, rank)
    assert vars(pmi17).keys() == vars(pmi18).keys()
    assert np.allclose(r17.precoder(pmi17), predicted.precoder(pmi18))
    assert r17.overhead_bits(pmi17) == predicted.overhead_bits(pmi18)
    assert pack(r17, pmi17) == pack(predicted, pmi18)
    restored = unpack(predicted, pack(predicted, pmi18), rank)
    assert np.allclose(predicted.precoder(restored), predicted.precoder(pmi18))


def test_configuration_and_restriction_guards():
    with pytest.raises(ValueError, match="R=2"):
        R18PredictedPortSelectionCodebook(ANT, N3=8, param_combination=2, R=2)
    with pytest.raises(ValueError, match="R must"):
        R18PredictedPortSelectionCodebook(ANT, N3=8, R=3)
    cbk = R18PredictedPortSelectionCodebook(
        ANT, N3=8, rank_restriction=np.array([1, 0, 1, 1])
    )
    with pytest.raises(ValueError, match="prohibited"):
        cbk.select(random_channel(), rank=2)
    assert cbk.N4 == 1


def test_harness_integration():
    cbk = R18PredictedPortSelectionCodebook(ANT, N3=8, param_combination=5)
    result = evaluate(
        cbk,
        RandomRayChannel(ANT, N3=8, n_rx=2),
        rank=2,
        n_drops=2,
        snr_db=[10],
    )
    assert 0 <= result.subspace_sgcs <= 1
