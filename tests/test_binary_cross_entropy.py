from src.losses import bernoulli_nll


def test_bernoulli_nll_rewards_good_prediction():
    good = bernoulli_nll(y=1.0, p=0.95)
    bad = bernoulli_nll(y=1.0, p=0.05)
    assert good < bad
