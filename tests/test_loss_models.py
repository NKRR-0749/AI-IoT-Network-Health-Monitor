import random

import pytest

from iotmon.impair.models import BernoulliLoss, GilbertElliottLoss, make_loss_model


def run(model, n=200_000):
    drops = [model.drop() for _ in range(n)]
    bursts, cur = [], 0
    for d in drops:
        if d:
            cur += 1
        elif cur:
            bursts.append(cur)
            cur = 0
    return sum(drops) / n, (sum(bursts) / len(bursts) if bursts else 0)


def test_bernoulli_mean_loss():
    loss, _ = run(BernoulliLoss(0.05, random.Random(1)))
    assert loss == pytest.approx(0.05, abs=0.005)


def test_gilbert_elliott_mean_loss_and_burst_length():
    loss, burst = run(GilbertElliottLoss(0.05, 10, random.Random(2)))
    assert loss == pytest.approx(0.05, abs=0.01)
    assert burst == pytest.approx(10, rel=0.15)


def test_factory_picks_model():
    assert isinstance(make_loss_model(5, 0), BernoulliLoss)
    assert isinstance(make_loss_model(5, 8), GilbertElliottLoss)
    assert not make_loss_model(0).drop()
