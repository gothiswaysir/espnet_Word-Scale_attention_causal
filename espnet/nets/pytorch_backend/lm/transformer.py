"""Transformer language model."""

from typing import Any
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.lm_interface import LMInterface
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_wordscale import Encoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.scorer_interface import BatchScorerInterface


class TransformerLM(nn.Module, LMInterface, BatchScorerInterface):
    """Transformer language model."""

    @staticmethod
    def add_arguments(parser):
        """Add arguments to command line argument parser."""
        parser.add_argument(
            "--layer", type=int, default=4, help="Number of hidden layers"
        )
        parser.add_argument(
            "--unit",
            type=int,
            default=1024,
            help="Number of hidden units in feedforward layer",
        )
        parser.add_argument(
            "--att-unit",
            type=int,
            default=256,
            help="Number of hidden units in attention layer",
        )
        parser.add_argument(
            "--embed-unit",
            type=int,
            default=128,
            help="Number of hidden units in embedding layer",
        )
        parser.add_argument(
            "--head", type=int, default=2, help="Number of multi head attention"
        )
        parser.add_argument(
            "--dropout-rate", type=float, default=0.5, help="dropout probability"
        )
        parser.add_argument(
            "--pos-enc",
            default="sinusoidal",
            choices=["sinusoidal", "none"],
            help="positional encoding",
        )
        return parser

    def __init__(self, n_vocab, args):
        """Initialize class.

        Args:
            n_vocab (int): The size of the vocabulary
            args (argparse.Namespace): configurations. see py:method:`add_arguments`

        """
        nn.Module.__init__(self)
        if args.pos_enc == "sinusoidal":
            pos_enc_class = PositionalEncoding
        elif args.pos_enc == "none":

            def pos_enc_class(*args, **kwargs):
                return nn.Sequential()  # indentity

        else:
            raise ValueError(f"unknown pos-enc option: {args.pos_enc}")

        self.embed = nn.Embedding(n_vocab, args.embed_unit)
        self.encoder = Encoder(
            idim=args.embed_unit,
            attention_dim=args.att_unit,
            attention_heads=args.head,
            linear_units=args.unit,
            num_blocks=args.layer,
            dropout_rate=args.dropout_rate,
            input_layer="linear",
            pos_enc_class=pos_enc_class,
        )
        self.decoder = nn.Linear(args.att_unit, n_vocab)

    def _target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != 0
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def _word_target_mask(self, ys_in_pad, aver_mask):
        ys_mask = ys_in_pad != 0
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        word_m = (aver_mask!=0) | m
        return ys_mask.unsqueeze(-2) & word_m

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, aver_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute LM loss value from buffer sequences.

        Args:
            x (torch.Tensor): Input ids. (batch, len)
            t (torch.Tensor): Target ids. (batch, len)

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple of
                loss to backward (scalar),
                negative log-likelihood of t: -log p(t) (scalar) and
                the number of elements in x (scalar)

        Notes:
            The last two return values are used
            in perplexity: p(t)^{-n} = exp(-log p(t) / n)

        """
        xm = x != 0
        h, _ = self.encoder(self.embed(x), self._target_mask(x), \
                            aver_mask)
        y = self.decoder(h)
        loss = F.cross_entropy(y.view(-1, y.shape[-1]), t.view(-1), reduction="none")

        # To print the loss of one single batch
        # import json
        # args_dict = "/xinyuan/work_lm_related/librispeech_100_base/data/lang_char/train_960_bpe5000_units.txt"
        # json_file_path = "/xinyuan/work_lm_related/librispeech_100_base/exp/train_transformerlm_pytorch_Word-Scale_attention_causal_2_bpe5000/onebatchtest_loss.json"
        #
        # with open(args_dict, "rb") as f:
        #     dictionary = f.readlines()
        # char_list = [entry.decode("utf-8").split(" ")[0] for entry in dictionary]
        # char_list.insert(0, "<blank>")
        # char_list.append("<eos>")
        # dict = {i: x for i, x in enumerate(char_list)}
        #
        # json_file = open(json_file_path, mode='w')
        # utts = []
        # for i in range(x.shape[0]):
        #     if False in xm[i]:
        #         pad_index = xm[i].numpy().tolist().index(False)
        #     else:
        #         pad_index = len(xm[i])
        #     input_text = ' '.join([dict.get(a) for a in x[i][:pad_index].numpy().tolist()])
        #     label_text = ' '.join([dict.get(a) for a in t[i][:pad_index].numpy().tolist()])
        #
        #     loss_each = F.cross_entropy(y[i], t[i], reduction="none").numpy().tolist()[:pad_index]
        #     loss_each_format = [float('{:.4f}'.format(j)) for j in loss_each]
        #     dict_save = {"input_text": input_text, "label_text": label_text, "loss_each": loss_each_format}
        #     utts.append(dict_save.copy())
        # json.dump(utts, json_file, indent=4)
        # json_file.close()

        mask = xm.to(dtype=loss.dtype)
        logp = loss * mask.view(-1)
        logp = logp.sum()
        count = mask.sum()
        return logp / count, logp, count

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """Score new token.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys.

        Returns:
            tuple[torch.Tensor, Any]: Tuple of
                torch.float32 scores for next token (n_vocab)
                and next state for ys

        """
        y = y.unsqueeze(0)
        h, _, cache = self.encoder.forward_one_step(
            self.embed(y), self._target_mask(y), cache=state
        )
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1).squeeze(0)
        return logp, cache

    # batch beam search API (see BatchScorerInterface)
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.encoder.encoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        # batch decoding
        h, _, states = self.encoder.forward_one_step(
            self.embed(ys), self._target_mask(ys), cache=batch_state
        )
        h = self.decoder(h[:, -1])
        logp = h.log_softmax(dim=-1)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list
