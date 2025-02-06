import numpy as np
import torch

from agilerl.utils.sampling_utils import (
    always_terminate,
    get_relevant_kvs,
    map_decoder_kvs,
    pad_sequence,
    process_logits,
    select_batch_idxs,
    update_decoder_kvs,
    update_kvs,
)


# The function returns a tensor with the same shape as x, but with the elements indexed by idxs in the first dimension.
def test_same_shape_with_indexed_elements():
    x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    idxs = torch.tensor([0, 2])
    expected_output = torch.tensor([[1, 2, 3], [7, 8, 9]])
    assert torch.all(select_batch_idxs(x, idxs) == expected_output)


# The function pads a sequence with a given value up to a certain length.
def test_pad_sequence_pads_sequence_with_given_value():
    seq = torch.tensor([1, 2, 3])
    to_len = 5
    val = 0
    device = "cpu"
    dim = 0

    result = pad_sequence(seq, to_len, val, device, dim)

    expected = torch.tensor([1, 2, 3, 0, 0])
    assert torch.all(torch.eq(result, expected))


# The function updates the key-value store (kvs) with the updated key-value store (updated_kvs) for a given index (idx) and chosen length (lens_chosen).
def test_update_kvs_with_updated_kvs():
    kvs = np.array(
        [
            [
                [
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                    [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
                ]
            ]
        ]
    )
    updated_kvs = np.array(
        [
            [
                [
                    [[[17, 18], [19, 20]], [[21, 22], [23, 24]]],
                    [[[25, 26], [27, 28]], [[29, 30], [31, 32]]],
                ]
            ]
        ]
    )
    lens_chosen = slice(0, None)
    idx = 0
    expected_result = np.array(
        [
            [[[17, 18], [3, 4]], [[21, 22], [7, 8]]],
            [[[25, 26], [11, 12]], [[29, 30], [15, 16]]],
        ]
    )

    result = update_kvs(kvs, updated_kvs, lens_chosen, idx)

    assert np.all(result == expected_result)


# The function updates the key-value store (kvs) with the updated key-value store (updated_kvs) for a given index (idx) and chosen length (lens_chosen).
def test_update_decoder_kvs_with_updated_kvs():
    kvs = np.array(
        [
            [
                [
                    [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                    [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
                ]
            ]
        ]
    )
    updated_kvs = np.array(
        [
            [
                [
                    [[[17, 18], [19, 20]], [[21, 22], [23, 24]]],
                    [[[25, 26], [27, 28]], [[29, 30], [31, 32]]],
                ]
            ]
        ]
    )
    lens_chosen = slice(0, None)
    idx = 0
    expected_result = np.array(
        [
            [[[17, 18], [3, 4]], [[21, 22], [7, 8]]],
            [[[25, 26], [11, 12]], [[29, 30], [15, 16]]],
        ]
    )

    result = update_decoder_kvs(kvs, updated_kvs, lens_chosen, idx)

    assert np.all(result == expected_result)


# The function returns the expected output when given valid inputs.
def test_get_relevant_kvs():
    kvs = (
        (
            torch.tensor([[[[[[1, 2, 3], [4, 5, 6]]]]]]),
            torch.tensor([[[[[[7, 8, 9], [10, 11, 12]]]]]]),
        ),
        (
            torch.tensor([[[[[[13, 14, 15], [16, 17, 18]]]]]]),
            torch.tensor([[[[[[19, 20, 21], [22, 23, 24]]]]]]),
        ),
    )
    lens_chosen = torch.tensor([0])
    idx = 2
    expected_output = (
        (
            torch.tensor([[[[[[1, 2, 3], [4, 5, 6]]]]]]),
            torch.tensor([[[[[[7, 8, 9], [10, 11, 12]]]]]]),
        ),
        (
            torch.tensor([[[[[[13, 14, 15], [16, 17, 18]]]]]]),
            torch.tensor([[[[[[19, 20, 21], [22, 23, 24]]]]]]),
        ),
    )

    assert str(get_relevant_kvs(kvs, lens_chosen, idx)) == str(expected_output)


def test_map_decoder_kvs():
    kvs = (
        (
            torch.tensor([[[[[[1, 2, 3], [4, 5, 6]]]]]]),
            torch.tensor([[[[[[7, 8, 9], [10, 11, 12]]]]]]),
        ),
        (
            torch.tensor([[[[[[13, 14, 15], [16, 17, 18]]]]]]),
            torch.tensor([[[[[[19, 20, 21], [22, 23, 24]]]]]]),
        ),
    )
    lens_chosen = torch.tensor([0])
    kvs = map_decoder_kvs(lambda x: select_batch_idxs(x, lens_chosen), kvs)

    expected_output = (
        (
            torch.tensor([[[[[[1, 2, 3], [4, 5, 6]]]]]]),
            torch.tensor([[[[[[7, 8, 9], [10, 11, 12]]]]]]),
        ),
        (
            torch.tensor([[[[[[13, 14, 15], [16, 17, 18]]]]]]),
            torch.tensor([[[[[[19, 20, 21], [22, 23, 24]]]]]]),
        ),
    )

    assert str(kvs) == str(expected_output)


def test_process_logits():
    logits = torch.randn(8, 4, 8, 4)

    out_logits = process_logits(logits, top_k=2, top_p=2)

    assert logits.shape == out_logits.shape


def test_always_terminate():
    assert always_terminate(np.array([0]))
