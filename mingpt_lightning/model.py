"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""
import math
from typing import Tuple

import einops
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.nn import functional as F


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        )


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
        )
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x: Tensor) -> Tensor:
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = map(
            lambda t: einops.rearrange(
                t, 'B T (nh hs) -> B nh T hs', nh=self.n_head, hs=C // self.n_head
            ),
            (q, k, v),
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = einops.rearrange(
            y, 'B nh T hs -> B T (nh hs)'
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(
        self,
        n_embd: int,
        n_head: int,
        block_size: int,
        attn_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, attn_pdrop, resid_pdrop)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            {
                'c_fc': nn.Linear(n_embd, 4 * n_embd),
                'c_proj': nn.Linear(4 * n_embd, n_embd),
                'act': NewGELU(),
                'dropout': nn.Dropout(resid_pdrop),
            }
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x


class GPT(LightningModule):
    """GPT Language Model"""

    def __init__(
        self,
        n_layer: int,
        n_head: int,
        n_embd: int,
        vocab_size: int,
        block_size: int,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size

        self.transformer = nn.ModuleDict(
            {
                'wte': nn.Embedding(vocab_size, n_embd),
                'wpe': nn.Embedding(block_size, n_embd),
                'drop': nn.Dropout(embd_pdrop),
                'h': nn.ModuleList(
                    [
                        Block(n_embd, n_head, block_size, attn_pdrop, resid_pdrop)
                        for _ in range(n_layer)
                    ]
                ),
                'ln_f': nn.LayerNorm(n_embd),
            }
        )
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(
        cls,
        model_type: str,
        n_layer: int,
        n_head: int,
        n_embd: int,
        vocab_size: int,
        block_size: int,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        learning_rate: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.1,
    ):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        assert vocab_size == 50257  # openai's model vocabulary
        assert block_size == 1024  # openai's model block_size
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        model = GPT(
            n_layer,
            n_head,
            n_embd,
            vocab_size,
            block_size,
            embd_pdrop,
            resid_pdrop,
            attn_pdrop,
            learning_rate,
            betas,
            weight_decay,
        )
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')]  # ignore these
        transposed = [
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight',
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def forward(self, idx: Tensor, pos: Tensor) -> Tensor:  # type: ignore
        b, t = idx.size()
        assert (
            t <= self.block_size
        ), f'Cannot forward sequence of length {t}, block size is only {self.block_size}'

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits

    def training_step(self, batch: Tuple, batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        return self._shared_step(batch)

    def validation_step(self, batch: Tuple, batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        return self._shared_step(batch)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        if self.weight_decay > 0.0:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear,)
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                    # random note: because named_modules and named_parameters are recursive
                    # we will see the same tensors p many many times. but doing it this way
                    # allows us to know which parent module any tensor p belongs to...
                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters()}
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert (
                len(inter_params) == 0
            ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
            assert (
                len(param_dict.keys() - union_params) == 0
            ), "parameters %s were not separated into either decay/no_decay set!" % (
                str(param_dict.keys() - union_params),
            )

            # create the pytorch optimizer object
            optim_groups = [
                {
                    "params": [param_dict[pn] for pn in sorted(list(decay))],
                    "weight_decay": self.weight_decay,
                },
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
        else:
            optim_groups = self.parameters()
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=self.betas)
        return optimizer

    def _shared_step(self, batch: Tuple) -> Tensor:
        # Assume batch is tuple where [0] is input, [1] is positions, [2] is targets
        logits = self(batch[0], batch[1])

        logits = einops.rearrange(logits, 'b s v -> (b s) v')
        targets = einops.rearrange(batch[2], 'b s -> (b s)')

        loss = F.cross_entropy(logits, targets, ignore_index=-1)

        return loss
