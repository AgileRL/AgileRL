import numpy as np
import pytest
import torch

from agilerl.modules.bert import (
    EvolvableBERT,
    PositionalEncoding,
    TokenEmbedding,
    _canonical_mask,
    _none_or_dtype,
)


#### TESTING EVOLVABLE BERT CLASS ####
class TestEvolvableBERTInit:
    def test_evolvable_bert_init_default(self):
        encoder_layers = [4, 4]
        decoder_layers = [4, 4]

        BERT = EvolvableBERT(encoder_layers, decoder_layers)

        assert BERT.encoder_layers == encoder_layers
        assert BERT.decoder_layers == decoder_layers
        assert BERT.end2end is True
        assert BERT.src_vocab_size == 10837
        assert BERT.tgt_vocab_size == 10837
        assert BERT.encoder_norm is True
        assert BERT.decoder_norm is True
        assert BERT.d_model == 512
        assert BERT.n_head == 8
        assert BERT.dropout == 0.1
        assert BERT.activation == "relu"
        assert BERT.layer_norm_eps == 1e-5
        assert BERT.batch_first is False
        assert BERT.norm_first is False
        assert BERT.max_encoder_layers == 12
        assert BERT.max_decoder_layers == 12
        assert BERT.device == "cpu"

    def test_evolvable_bert_init_no_e2e(self):
        encoder_layers_list = [[], [4, 4]]
        decoder_layers_list = [[4], [4, 4]]

        for encoder_layers, decoder_layers in zip(
            encoder_layers_list,
            decoder_layers_list,
            strict=False,
        ):
            BERT = EvolvableBERT(encoder_layers, decoder_layers, end2end=False)

            assert BERT.encoder_layers == encoder_layers
            assert BERT.decoder_layers == decoder_layers
            assert BERT.end2end is False
            assert isinstance(BERT.wte, TokenEmbedding)
            assert isinstance(BERT.wpe, PositionalEncoding)


class TestEvolvableBERTGenerateSquareSubsequentMask:
    def test_generate_square_subsequent_mask(self):
        model = EvolvableBERT([1], [1])
        mask = model.generate_square_subsequent_mask(2)
        assert str(mask) == str(torch.Tensor([[0.0, -np.inf], [0.0, 0.0]]))


class TestEvolvableBERTCreateMask:
    def test_create_mask(self):
        src = torch.zeros(2, 2)
        tgt = torch.zeros(4, 4)
        pad_idx = 1

        model = EvolvableBERT([1], [1])
        src_mask, tgt_mask, src_pm, tgt_pm = model.create_mask(src, tgt, pad_idx)

        assert torch.equal(src_mask, torch.Tensor([[False, False], [False, False]]))
        assert str(tgt_mask) == str(
            torch.Tensor(
                [
                    [0.0, -np.inf, -np.inf, -np.inf],
                    [0.0, 0.0, -np.inf, -np.inf],
                    [0.0, 0.0, 0.0, -np.inf],
                    [0.0, 0.0, 0.0, 0.0],
                ],
            ),
        )
        assert not torch.all(src_pm)
        assert not torch.all(tgt_pm)


class TestEvolvableBERTForward:
    def test_forward(self):
        src = torch.LongTensor([[1, 2, 4, 5]])
        tgt = torch.LongTensor([[1, 2, 4, 5]])

        model = EvolvableBERT([1], [1])
        mask = torch.zeros(1, 1)
        output = model(src, tgt, src_mask=mask, is_causal=None)

        assert output.shape == (1, 4, 10837)


class TestEvolvableBERTCountParameters:
    def test_count_parameters(self):
        model = EvolvableBERT([1], [1])
        params = model.count_parameters()
        assert params == 19818583


class TestEvolvableBERTAddEncoderLayer:
    def test_add_encoder_layer(self):
        model = EvolvableBERT([1], [1], encoder_norm=False)
        initial_n_layer = len(model.encoder_layers)
        model.add_encoder_layer()
        assert len(model.encoder_layers) == initial_n_layer + 1
        assert len(model.encoder) == initial_n_layer + 1


class TestEvolvableBERTAddDecoderLayer:
    def test_add_decoder_layer(self):
        model = EvolvableBERT([1], [1], decoder_norm=False)
        initial_n_layer = len(model.decoder_layers)
        model.add_decoder_layer()
        assert len(model.decoder_layers) == initial_n_layer + 1
        assert len(model.decoder) == initial_n_layer + 1


class TestEvolvableBERTRemoveEncoderLayer:
    def test_remove_encoder_layer(self):
        model = EvolvableBERT([1, 1], [1, 1], encoder_norm=False)
        initial_n_layer = len(model.encoder_layers)
        model.remove_encoder_layer()
        assert len(model.encoder_layers) == initial_n_layer - 1
        assert len(model.encoder) == initial_n_layer - 1


class TestEvolvableBERTRemoveDecoderLayer:
    def test_remove_decoder_layer(self):
        model = EvolvableBERT([1, 1], [1, 1], decoder_norm=False)
        initial_n_layer = len(model.decoder_layers)
        model.remove_decoder_layer()
        assert len(model.decoder_layers) == initial_n_layer - 1
        assert len(model.decoder) == initial_n_layer - 1


class TestEvolvableBERTAddNode:
    @pytest.mark.parametrize(
        "network, hidden_layer, numb_new_nodes",
        [
            (None, 0, None),
            ("encoder", 0, None),
            ("encoder", None, None),
            ("decoder", 0, None),
            ("decoder", None, None),
        ],
    )
    def test_add_node(self, network, hidden_layer, numb_new_nodes):
        model = EvolvableBERT([32], [32])
        nodes_dict = model.add_node(network, hidden_layer, numb_new_nodes)

        if network is not None:
            assert nodes_dict["network"] == network
        else:
            assert nodes_dict["network"] in ["encoder", "decoder"]
        assert nodes_dict["hidden_layer"] == 0
        assert nodes_dict["numb_new_nodes"] in [16, 32, 64]

        if nodes_dict["network"] == "encoder":
            assert (
                model.encoder_layers[nodes_dict["hidden_layer"]]
                == 32 + nodes_dict["numb_new_nodes"]
            )
        else:
            assert (
                model.decoder_layers[nodes_dict["hidden_layer"]]
                == 32 + nodes_dict["numb_new_nodes"]
            )


class TestEvolvableBERTRemoveNode:
    @pytest.mark.parametrize(
        "network, hidden_layer, numb_new_nodes",
        [
            (None, 0, None),
            ("encoder", 0, None),
            ("encoder", None, None),
            ("decoder", 0, None),
            ("decoder", None, None),
        ],
    )
    def test_remove_node(self, network, hidden_layer, numb_new_nodes):
        model = EvolvableBERT([256], [256])
        nodes_dict = model.remove_node(network, hidden_layer, numb_new_nodes)

        if network is not None:
            assert nodes_dict["network"] == network
        else:
            assert nodes_dict["network"] in ["encoder", "decoder"]
        assert nodes_dict["hidden_layer"] == 0
        assert nodes_dict["numb_new_nodes"] in [16, 32, 64]

        if nodes_dict["network"] == "encoder":
            assert (
                model.encoder_layers[nodes_dict["hidden_layer"]]
                == 256 - nodes_dict["numb_new_nodes"]
            )
        else:
            assert (
                model.decoder_layers[nodes_dict["hidden_layer"]]
                == 256 - nodes_dict["numb_new_nodes"]
            )


class TestEvolvableBERTClone:
    def test_clone(self):
        model = EvolvableBERT([1], [1])
        old_init_dict = model.init_dict
        clone = model.clone()

        assert clone.init_dict == old_init_dict


class TestEvolvableBERTEncode:
    def test_encode(self):
        src = torch.LongTensor([[1, 2, 4, 5]])

        model = EvolvableBERT([4, 4], [4, 4], norm_first=True, batch_first=True)
        src_mask = None
        kp_mask = torch.zeros(1, 4)

        model.eval()
        # Set first layer to eval
        encoder_output, _ = model.encode(
            src,
            src_mask=src_mask,
            src_key_padding_mask=kp_mask,
        )
        assert encoder_output.shape == (1, 4, 512)


class TestEvolvableBERTCheckEncoderSparsityFastPath:
    def test_check_sparsity_fast_path(self):
        src = torch.LongTensor([[1, 2, 4, 5]])

        model = EvolvableBERT([4, 4], [4, 4], norm_first=True, batch_first=True)
        src_mask = None
        kp_mask = torch.zeros(1, 4)

        if model.end2end:
            src = model.positional_encoding(model.src_tok_emb(src))

        # Encoder forward pass preparation
        src_key_padding_mask = _canonical_mask(
            mask=kp_mask,
            mask_name="src_key_padding_mask",
            other_type=_none_or_dtype(src_mask),
            other_name="mask",
            target_type=src.dtype,
        )
        encoder_output = src
        first_layer = model.encoder[model.encoder_keys[0]]
        str_first_layer = "self.net[0]"
        src_key_padding_mask_for_layers = src_key_padding_mask
        (
            encoder_output,
            convert_to_nested,
            src_key_padding_mask_for_layers,
        ) = model.check_encoder_sparsity_fast_path(
            src,
            encoder_output,
            first_layer,
            str_first_layer,
            src_mask,
            src_key_padding_mask,
            src_key_padding_mask_for_layers,
        )
        assert convert_to_nested is True
        assert src_key_padding_mask_for_layers is None


class TestCanonicalMask:
    @pytest.mark.parametrize(
        "mask, mask_name, other_type, other_name, target_type, check_other, error",
        [
            (torch.zeros(4, 1), "mask", "na", "other", "int", True, None),
            (torch.zeros(4, 1).bool(), "mask", "na", "other", torch.float, True, None),
            (
                torch.ones(4, 1).int() * 2,
                "mask",
                "na",
                "other",
                "int",
                True,
                AssertionError,
            ),
        ],
    )
    def test_canconical_mask_failures(
        self,
        mask,
        mask_name,
        other_type,
        other_name,
        target_type,
        check_other,
        error,
    ):
        if error is not None:
            with pytest.raises(error):
                _canonical_mask(
                    mask,
                    mask_name,
                    other_type,
                    other_name,
                    target_type,
                    check_other,
                )
        else:
            mask = _canonical_mask(
                mask,
                mask_name,
                other_type,
                other_name,
                target_type,
                check_other,
            )

            assert mask.shape == (4, 1)


#### TESTING POSITIONAL ENCODING CLASS ####
class TestPositionalEncodingForward:
    def test_pos_encoding(self):
        input_tensor = torch.LongTensor([[1, 2, 4, 5]])
        pos = PositionalEncoding(10, 3)
        enc = pos(input_tensor)

        assert enc.shape == (1, 4, 3)


#### TESTING TOKEN EMBEDDING CLASS ####
class TestTokenEmbeddingForward:
    def test_tok_embedding(self):
        input_tensor = torch.LongTensor([[1, 2, 4, 5]])
        tok = TokenEmbedding(10, 3)
        emb = tok(input_tensor)

        assert emb.shape == (1, 4, 3)


class TestEvolvableBERTActivation:
    def test_bert_activation_setter(self):
        model = EvolvableBERT([4], [4])
        model.activation = "gelu"
        assert model._activation == "gelu"


class TestEvolvableBERTDecode:
    def test_bert_decode_with_decoder_norm(self):
        src = torch.LongTensor([[1, 2, 4, 5]])
        tgt = torch.LongTensor([[1, 2, 4, 5]])
        model = EvolvableBERT([4, 4], [4, 4], encoder_norm=True, decoder_norm=True)
        model.eval()
        memory = model.encode(src)[0]
        decoder_out, _ = model.decode(tgt, memory)
        assert decoder_out.shape == (1, 4, 512)


#### TESTING NONE OR DTYPE FUNCTION ####
class TestNoneOrDtype:
    def test_non_or_dtype(self):
        func_input = None
        func_output = _none_or_dtype(func_input)
        assert func_output is None

        func_input = torch.Tensor([0])
        func_output = _none_or_dtype(func_input)
        assert func_output == torch.float32

        func_input = 0
        with pytest.raises(RuntimeError):
            _none_or_dtype(func_input)
