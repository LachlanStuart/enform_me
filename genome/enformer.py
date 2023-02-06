# torch.jit.script-compatible version of https://github.com/lucidrains/enformer-pytorch
# MIT License

# Copyright (c) 2021 Phil Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential

# from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from enformer_pytorch.data import str_to_one_hot, seq_indices_to_one_hot

from enformer_pytorch.config_enformer import EnformerConfig

from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.generation import GenerationConfig

# constants

SEQUENCE_LENGTH = 196_608
TARGET_LENGTH = 896

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def map_values(fn, d):
    return {key: fn(values) for key, values in d.items()}


def exponential_linspace_int(start, end, num, divisible_by=1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


# losses and metrics


def poisson_loss(pred, target):
    return (pred - target * log(pred)).mean()


def pearson_corr_coef(x, y, dim=1, reduce_dims=(-1,)):
    x_centered = x - x.mean(dim=dim, keepdim=True)
    y_centered = y - y.mean(dim=dim, keepdim=True)
    return F.cosine_similarity(x_centered, y_centered, dim=dim).mean(dim=reduce_dims)


# relative positional encoding functions


def get_positional_features_exponential(positions, features: int, seq_len: int, min_half_life: float = 3.0) -> Tensor:
    max_range = math.log(seq_len) / math.log(2.0)
    half_life = 2 ** torch.linspace(min_half_life, max_range, features, device=positions.device)
    half_life = half_life[None, ...]
    positions = positions.abs()[..., None]
    return torch.exp(-math.log(2.0) / half_life * positions)


def get_positional_features_central_mask(positions, features: int) -> Tensor:
    center_widths = 2 ** torch.arange(1, features + 1, device=positions.device).float()
    center_widths = center_widths - 1
    return (center_widths[None, ...] > positions.abs()[..., None]).float()


def gamma_pdf(x, concentration, rate):
    log_unnormalized_prob = torch.xlogy(concentration - 1.0, x) - rate * x
    log_normalization = torch.lgamma(concentration) - concentration * torch.log(rate)
    return torch.exp(log_unnormalized_prob - log_normalization)


def get_positional_features_gamma(positions, features: int, seq_len: int, eps: float = 1e-8) -> Tensor:
    stddev = seq_len / (2 * features)
    start_mean = seq_len / features

    mean = torch.linspace(start_mean, seq_len, features, device=positions.device)
    mean = mean[None, ...]
    concentration = (mean / stddev) ** 2
    rate = mean / stddev**2
    probabilities = gamma_pdf(positions.float().abs()[..., None], concentration, rate)
    probabilities = probabilities + eps
    outputs = probabilities / torch.amax(probabilities, dim=-1, keepdim=True)
    return outputs


def get_positional_embed(seq_len: int, feature_size: int, device: torch.device):
    distances = torch.arange(-seq_len + 1, seq_len, device=device)

    # feature_functions = [
    #     get_positional_features_exponential,
    #     get_positional_features_central_mask,
    #     get_positional_features_gamma,
    # ]

    num_components = 3 * 2

    if (feature_size % num_components) != 0:
        raise ValueError(f"feature size is not divisible by number of components ({num_components})")

    num_basis_per_class = feature_size // num_components

    embeddings = torch.cat(
        [
            get_positional_features_exponential(distances, num_basis_per_class, seq_len),
            get_positional_features_central_mask(distances, num_basis_per_class),
            get_positional_features_gamma(distances, num_basis_per_class, seq_len),
        ],
        dim=-1,
    )
    embeddings = torch.cat((embeddings, torch.sign(distances)[..., None] * embeddings), dim=-1)
    return embeddings


def relative_shift(x):
    to_pad = torch.zeros_like(x[..., :1])
    x = torch.cat((to_pad, x), dim=-1)
    _, h, t1, t2 = x.shape
    x = x.reshape(-1, h, t2, t1)
    x = x[:, :, 1:, :]
    x = x.reshape(-1, h, t1, t2 - 1)
    return x[..., : ((t2 + 1) // 2)]


# classes


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class GELU(nn.Module):
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


class AttentionPool(nn.Module):
    def __init__(self, dim, pool_size=2):
        super().__init__()
        self.pool_size = pool_size
        self.pool_fn = Rearrange("b d (n p) -> b d n p", p=pool_size)
        self.to_attn_logits = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x):
        b, _, n = x.shape
        remainder = n % self.pool_size
        needs_padding = remainder > 0

        if needs_padding:
            x = F.pad(x, (0, remainder), value=0.0)
            mask = torch.zeros((b, 1, n), dtype=torch.bool, device=x.device)
            mask = F.pad(mask, (0, remainder), value=1.0)

            x = self.pool_fn(x)
            logits = self.to_attn_logits(x)

            if logits.dtype == torch.float16:
                mask_value = -65504.0
            elif logits.dtype == torch.float32:
                mask_value = -3.4028234663852886e38
            elif logits.dtype == torch.float64:
                mask_value = -1.7976931348623157e308
            else:
                mask_value = -3.3895313892515355e38

            logits = logits.masked_fill(self.pool_fn(mask), mask_value)
        else:
            x = self.pool_fn(x)
            logits = self.to_attn_logits(x)

        attn = logits.softmax(dim=-1)

        return (x * attn).sum(dim=-1)


class TargetLengthCrop(nn.Module):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def forward(self, x):
        seq_len, target_len = x.shape[-2], self.target_length

        if target_len == -1:
            return x

        if seq_len < target_len:
            raise ValueError(f"sequence length {seq_len} is less than target length {target_len}")

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return x

        return x[:, -trim:trim]


def ConvBlock(dim, dim_out=None, kernel_size=1):
    return nn.Sequential(
        nn.BatchNorm1d(dim), GELU(), nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding=kernel_size // 2)
    )


# attention classes


class Attention(nn.Module):
    def __init__(self, dim, *, num_rel_pos_features, heads=8, dim_key=64, dim_value=64, dropout=0.0, pos_dropout=0.0):
        super().__init__()
        self.scale = dim_key**-0.5
        self.heads = heads

        self.to_q = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_key * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_value * heads, bias=False)

        self.to_out = nn.Linear(dim_value * heads, dim)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

        # relative positional encoding

        self.num_rel_pos_features = num_rel_pos_features

        self.to_rel_k = nn.Linear(num_rel_pos_features, dim_key * heads, bias=False)
        self.rel_content_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))
        self.rel_pos_bias = nn.Parameter(torch.randn(1, heads, 1, dim_key))

        # dropouts

        self.pos_dropout = nn.Dropout(pos_dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self.bnhd_to_bhnd = Rearrange("b n (h d) -> b h n d", h=self.heads)
        self.nhd_to_hnd = Rearrange("n (h d) -> h n d", h=self.heads)
        self.bhnd_to_bnhd = Rearrange("b h n d -> b n (h d)")

    def forward(self, x):
        n = x.shape[-2]
        device = x.device

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # CHANGED: Remove lambda
        # q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))
        q = self.bnhd_to_bhnd(q)
        k = self.bnhd_to_bhnd(k)
        v = self.bnhd_to_bhnd(v)

        q = q * self.scale

        content_logits = torch.einsum("b h i d, b h j d -> b h i j", (q + self.rel_content_bias).float(), k.float())

        positions = get_positional_embed(n, self.num_rel_pos_features, device)
        positions = self.pos_dropout(positions)
        rel_k = self.to_rel_k(positions)

        rel_k = self.nhd_to_hnd(rel_k)
        rel_logits = torch.einsum("b h i d, h j d -> b h i j", (q + self.rel_pos_bias).float(), rel_k.float())
        rel_logits = relative_shift(rel_logits)

        logits = content_logits + rel_logits
        attn = logits.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn.float(), v.float())
        out = self.bhnd_to_bnhd(out)
        return self.to_out(out)


# main class


class Enformer(PreTrainedModel):
    config_class = EnformerConfig
    base_model_prefix = "enformer"

    # Override to remove type annotation
    # @property
    # def base_model(self):
    #     """
    #     `torch.nn.Module`: The main body of the model.
    #     """
    #     return self

    @staticmethod
    def from_hparams(**kwargs):
        return Enformer(EnformerConfig(**kwargs))

    def __init__(self, config):
        if isinstance(self, PreTrainedModel):
            super().__init__(config)
        else:
            nn.Module.__init__(self)
            self.config = config

        self.dim = config.dim
        half_dim = config.dim // 2
        twice_dim = config.dim * 2

        # create stem

        self.stem = nn.Sequential(
            nn.Conv1d(4, half_dim, 15, padding=7), Residual(ConvBlock(half_dim)), AttentionPool(half_dim, pool_size=2)
        )

        # create conv tower

        filter_list = exponential_linspace_int(
            half_dim, config.dim, num=(config.num_downsamples - 1), divisible_by=config.dim_divisible_by
        )
        filter_list = [half_dim, *filter_list]

        conv_layers = []
        for dim_in, dim_out in zip(filter_list[:-1], filter_list[1:]):
            conv_layers.append(
                nn.Sequential(
                    ConvBlock(dim_in, dim_out, kernel_size=5),
                    Residual(ConvBlock(dim_out, dim_out, 1)),
                    AttentionPool(dim_out, pool_size=2),
                )
            )

        self.conv_tower = nn.Sequential(*conv_layers)

        # transformer

        transformer = []
        for _ in range(config.depth):
            transformer.append(
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(config.dim),
                            Attention(
                                config.dim,
                                heads=config.heads,
                                dim_key=config.attn_dim_key,
                                dim_value=config.dim // config.heads,
                                dropout=config.attn_dropout,
                                pos_dropout=config.pos_dropout,
                                num_rel_pos_features=config.dim // config.heads,
                            ),
                            nn.Dropout(config.dropout_rate),
                        )
                    ),
                    Residual(
                        nn.Sequential(
                            nn.LayerNorm(config.dim),
                            nn.Linear(config.dim, config.dim * 2),
                            nn.Dropout(config.dropout_rate),
                            nn.ReLU(),
                            nn.Linear(config.dim * 2, config.dim),
                            nn.Dropout(config.dropout_rate),
                        )
                    ),
                )
            )

        self.transformer = nn.Sequential(*transformer)

        # target cropping

        self.target_length = config.target_length
        self.crop_final = TargetLengthCrop(config.target_length)

        # final pointwise

        self.final_pointwise = nn.Sequential(
            Rearrange("b n d -> b d n"),
            ConvBlock(filter_list[-1], twice_dim, 1),
            Rearrange("b d n -> b n d"),
            nn.Dropout(config.dropout_rate / 8),
            GELU(),
        )

        # create trunk sequential module
        self.bnd_to_bdn = Rearrange("b n d -> b d n")
        self.bdn_to_bnd = Rearrange("b d n -> b n d")
        self.unbatch_to_batch = Rearrange("... -> () ...")
        self.batch_to_unbatch = Rearrange("() ... -> ...")

        self._trunk = nn.Sequential(
            Rearrange("b n d -> b d n"),
            self.stem,
            self.conv_tower,
            Rearrange("b d n -> b n d"),
            self.transformer,
            self.crop_final,
            self.final_pointwise,
        )

        # create final heads for human and mouse

        self.add_heads(**config.output_heads)

        # use checkpointing on transformer trunk

        self.use_checkpointing = config.use_checkpointing

    # CHANGED: remove lambda, kwargs
    def add_heads(self, human=5313, mouse=1643):
        self.output_heads = {"human": 5313, "mouse": 1643}
        self._heads = nn.ModuleDict(
            {
                "human": nn.Sequential(nn.Linear(self.dim * 2, human), nn.Softplus()),
                "mouse": nn.Sequential(nn.Linear(self.dim * 2, mouse), nn.Softplus()),
            }
        )

    def set_target_length(self, target_length):
        crop_module = self._trunk[-2]
        crop_module.target_length = target_length

    @property
    def trunk(self):
        return self._trunk

    @property
    def heads(self):
        return self._heads

    # def trunk_checkpointed(self, x):
    #     x = self.bnd_to_bdn(x)
    #     x = self.stem(x)
    #     x = self.conv_tower(x)
    #     x = self.bdn_to_bnd(x)
    #     x = checkpoint_sequential(self.transformer, len(self.transformer), x)
    #     x = self.crop_final(x)
    #     x = self.final_pointwise(x)
    #     return x

    def forward(self, x):
        # if isinstance(x, list):
        #     x = str_to_one_hot(x)

        # elif x.dtype == torch.long:
        #     x = seq_indices_to_one_hot(x)
        x = x.float()
        # no_batch = x.ndim == 2

        # if no_batch:
        #     x = self.unbatch_to_batch(x)

        # if self.use_checkpointing:
        #     x = self.trunk_checkpointed(x)
        # else:
        x = self._trunk(x)

        # if no_batch:
        #     x = self.batch_to_unbatch(x)

        # if return_only_embeddings:
        return x

        # CHANGED: remove lambda
        # out = map_values(lambda fn: fn(x), self._heads)
        # out = {
        #     'human': self._heads['human'](x),
        #     'mouse': self._heads['mouse'](x),
        # }

        # if exists(head):
        #     # assert head in self._heads, f"head {head} not found"
        #     out = out[head]

        # if exists(target):
        #     # assert exists(head), "head must be passed in if one were to calculate loss directly with targets"

        #     if return_corr_coef:
        #         return pearson_corr_coef(out, target)

        #     return poisson_loss(out, target)

        # if return_embeddings:
        #     return out, x

        # return out


RawEnformer = type("RawEnformer", (nn.Module,), dict(Enformer.__dict__))


def from_pretrained(path, *args, **kwargs):
    model = Enformer.from_pretrained(path, *args, **kwargs)
    raw_model = RawEnformer(model.config)
    raw_model.load_state_dict(model.state_dict())
    return raw_model
