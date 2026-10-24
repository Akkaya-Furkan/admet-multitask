import numpy as np
import torch

from admet_multitask.model import Network, observed_loss, fit


def test_masked_loss_and_gradient():
    prediction = torch.tensor([1., 2., 3.], requires_grad=True)
    loss = observed_loss(prediction, torch.tensor([0., float("nan"), 1.]))
    assert loss.item() == 2.5
    loss.backward()
    assert prediction.grad[1] == 0
    assert torch.isfinite(prediction.grad).all()


def test_empty_task_loss_is_finite_zero():
    prediction = torch.tensor([1.], requires_grad=True)
    loss = observed_loss(prediction, torch.tensor([float("nan")]))
    loss.backward()
    assert loss.item() == 0 and prediction.grad.item() == 0


def test_classification_loss_masks_nan():
    loss = observed_loss(torch.zeros(2), torch.tensor([1., float("nan")]), True)
    np.testing.assert_allclose(loss.item(), np.log(2), rtol=1e-6)


def test_matched_trunk_initialization():
    torch.manual_seed(11)
    st = Network(input_dim=16)
    torch.manual_seed(11)
    mt = Network(input_dim=16)
    for a, b in zip(st.parameters(), mt.parameters()):
        assert torch.equal(a, b)


def test_training_live_path_deterministic_and_test_labels_unused():
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(90, 16)).astype(np.float32)
    y = rng.normal(size=(90, 3)).astype(np.float32)
    y[:, 2] = (y[:, 2] > 0).astype(float)
    split = np.array(["train"] * 60 + ["valid"] * 15 + ["test"] * 15)
    config = {"hidden": [8, 4], "learning_rate": .01, "weight_decay": .0001,
              "batch_size": 20, "max_epochs": 3, "patience": 2}
    a, info = fit(x, y, split, [0, 1, 2], 11, config)
    modified = y.copy()
    modified[split == "test"] = np.nan
    b, other = fit(x, modified, split, [0, 1, 2], 11, config)
    assert info == other
    for left, right in zip(a.parameters(), b.parameters()):
        assert torch.equal(left, right)
    st, _ = fit(x, y, split, [0], 11, config)
    assert st(torch.from_numpy(x[:2]), 0).shape == (2,)
