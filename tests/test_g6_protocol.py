"""Negative controls for frozen G6 data provenance and benchmark budget."""
import pytest
import torch
from scripts.g6_protocol import (PROTOCOL, SEEDS, TRAIN_STEPS, VAL_BATCHES,
                                  SEQ_LEN, ARMS, batch_seed, make_batch, protocol_sha256)


def test_frozen_protocol_shape_and_identity():
    assert len(SEEDS) == 3 and TRAIN_STEPS == 96
    assert SEQ_LEN == 16 and VAL_BATCHES == 8 and len(ARMS) == 4
    assert len(protocol_sha256()) == 64
    assert PROTOCOL['science_class'] == 'development_screen_no_sealed_test'


def test_train_validation_seed_disjoint():
    train = {batch_seed(s, i, phase='train') for s in SEEDS for i in range(TRAIN_STEPS)}
    validation = {batch_seed(s, i, phase='validation') for s in SEEDS for i in range(VAL_BATCHES)}
    assert train.isdisjoint(validation)
    assert len(train) == len(SEEDS) * TRAIN_STEPS
    assert len(validation) == len(SEEDS) * VAL_BATCHES


def test_query_value_is_hidden_from_input():
    x, target = make_batch(SEEDS[0], 0, phase='validation')
    assert x.shape == (16, SEQ_LEN, 3) and target.shape == (16,)
    assert torch.all(x[:, -1, 1] == 0)
    assert torch.all(x[:, -1, 2] == 1)
    assert torch.all(x[:, :-1, 2] == 0)


def test_repeatable_batches():
    one = make_batch(SEEDS[1], 5, phase='train')
    two = make_batch(SEEDS[1], 5, phase='train')
    assert all(torch.equal(a, b) for a, b in zip(one, two))


def test_scramble_changes_support_not_query_or_labels():
    x, y = make_batch(SEEDS[2], 3, phase='validation')
    scrambled, same_y = make_batch(SEEDS[2], 3, phase='validation', scramble=True)
    assert torch.equal(y, same_y)
    assert torch.equal(x[:, -1], scrambled[:, -1])
    assert torch.equal(x[:, :-1, 0], scrambled[:, :-1, 0])
    assert not torch.equal(x[:, :-1, 1], scrambled[:, :-1, 1])


@pytest.mark.parametrize('phase,index', [('train', TRAIN_STEPS), ('validation', VAL_BATCHES),
                                         ('other', 0), ('train', -1)])
def test_out_of_protocol_samples_fail_closed(phase, index):
    with pytest.raises(ValueError):
        make_batch(SEEDS[0], index, phase=phase)


def test_unregistered_seed_rejected():
    with pytest.raises(ValueError):
        make_batch(999, 0, phase='train')
