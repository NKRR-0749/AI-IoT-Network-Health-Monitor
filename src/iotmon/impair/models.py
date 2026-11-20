"""Block: packet-loss models.

Random (Bernoulli) loss: every packet is lost with the same probability p.
Real wireless loss is usually BURSTY: losses come in clumps. The classic model
for that is Gilbert-Elliott: a two-state Markov chain.

    GOOD state --p--> BAD state        (p = chance to enter a burst)
    BAD  state --r--> GOOD state       (r = chance to leave it; mean burst = 1/r)
    In BAD every packet is lost; in GOOD none is (the "Gilbert" special case).

Given a target mean loss L and mean burst length B:
    r = 1/B,   p = L*r / (1 - L)      (steady state: P(bad) = p/(p+r) = L)
netem's "loss gemodel p r 1 0" is the same model, so both backends match.
"""
import random


class BernoulliLoss:
    def __init__(self, loss: float, rng: random.Random | None = None):
        self.loss = max(0.0, min(1.0, loss))
        self.rng = rng or random.Random()

    def drop(self) -> bool:
        return self.loss > 0 and self.rng.random() < self.loss


class GilbertElliottLoss:
    def __init__(self, loss: float, burst_len: float, rng: random.Random | None = None):
        self.rng = rng or random.Random()
        loss = max(0.0, min(0.99, loss))
        self.r = 1.0 / max(1.0, burst_len)
        self.p = loss * self.r / (1.0 - loss) if loss > 0 else 0.0
        self.bad = False

    def drop(self) -> bool:
        if self.bad:
            if self.rng.random() < self.r:
                self.bad = False
        elif self.rng.random() < self.p:
            self.bad = True
        return self.bad

    def netem_args(self) -> str:
        return f"loss gemodel {100 * self.p:.4f}% {100 * self.r:.4f}% 100% 0%"


def make_loss_model(loss_pct: float, burst_len: float = 0, rng: random.Random | None = None):
    """burst_len <= 1 -> random loss; otherwise Gilbert-Elliott bursts of that mean length."""
    loss = loss_pct / 100.0
    if burst_len and burst_len > 1:
        return GilbertElliottLoss(loss, burst_len, rng)
    return BernoulliLoss(loss, rng)
