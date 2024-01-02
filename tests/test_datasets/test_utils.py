import pytest
import torch

from openstl.models import build_loss

def test_cross_entropy_loss():
    with pytest.raises(AssertionError):
        # use_sigmoid and use_soft could not be set simultaneously
        loss_cfg = dict(
            type='CrossEntropyLoss', use_sigmoid=True, use_soft=True)
        loss = build_loss(loss_cfg)

    # test ce_loss
    cls_score = torch.Tensor([[-1000, 1000], [100, -100]])
    label = torch.Tensor([0, 1]).long()
    class_weight = [0.3, 0.7]  # class 0 : 0.3, class 1 : 0.7
    weight = torch.tensor([0.6, 0.4])

    # test ce_loss without class weight
    loss_cfg = dict(type='CrossEntropyLoss', reduction='mean', loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(1100.))
    # test ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(640.))

    # test ce_loss with class weight
    loss_cfg = dict(
        type='CrossEntropyLoss',
        reduction='mean',
        loss_weight=1.0,
        class_weight=class_weight)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(370.))
    # test ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(208.))

    # test bce_loss
    cls_score = torch.Tensor([[-200, 100], [500, -1000], [300, -300]])
    label = torch.Tensor([[1, 0], [0, 1], [1, 0]])
    weight = torch.Tensor([0.6, 0.4, 0.5])
    class_weight = [0.1, 0.9]  # class 0: 0.1, class 1: 0.9
    pos_weight = [0.1, 0.2]

    # test bce_loss without class weight
    loss_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(300.))
    # test ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(130.))

    # test bce_loss with class weight
    loss_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='mean',
        loss_weight=1.0,
        class_weight=class_weight)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(176.667))
    # test bce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(74.333))

    # test bce loss with pos_weight
    loss_cfg = dict(
        type='CrossEntropyLoss',
        use_sigmoid=True,
        reduction='mean',
        loss_weight=1.0,
        pos_weight=pos_weight)
    loss = build_loss(loss_cfg)

    # test soft_ce_loss
    cls_score = torch.Tensor([[-1000, 1000], [100, -100]])
    label = torch.Tensor([[1.0, 0.0], [0.0, 1.0]])
    class_weight = [0.3, 0.7]  # class 0 : 0.3, class 1 : 0.7
    weight = torch.tensor([0.6, 0.4])

    # test soft_ce_loss without class weight
    loss_cfg = dict(
        type='CrossEntropyLoss',
        use_soft=True,
        reduction='mean',
        loss_weight=1.0)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(1100.))
    # test soft_ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(640.))

    # test soft_ce_loss with class weight
    loss_cfg = dict(
        type='CrossEntropyLoss',
        use_soft=True,
        reduction='mean',
        loss_weight=1.0,
        class_weight=class_weight)
    loss = build_loss(loss_cfg)
    assert torch.allclose(loss(cls_score, label), torch.tensor(370.))
    # test soft_ce_loss with weight
    assert torch.allclose(
        loss(cls_score, label, weight=weight), torch.tensor(208.))