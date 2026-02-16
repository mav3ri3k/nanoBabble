from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from config import Config


class Transformer(nnx.Module):
    def __init__(self, cfg: Config, *, rngs: nnx.Rngs):
        self.num_layers = cfg.num_layers
        self.embed = nnx.Embed(cfg.vocab_size, cfg.dim, rngs=rngs)
        self.pos_embed = nnx.Embed(cfg.ctx_len, cfg.dim, rngs=rngs)
        self.blocks = nnx.List(
            [
            TransformerBlock(cfg=cfg, layer_id=layer_id, n_layers=cfg.num_layers, rngs=rngs)
            for layer_id in range(cfg.num_layers)
            ]
        )
        self.out_ln = nnx.RMSNorm(num_features=cfg.dim, rngs=rngs)

    def __call__(self, token_ids_bl: jnp.ndarray) -> jnp.ndarray:
        x_bld = self.embed(token_ids_bl)
        pos_ids = jnp.arange(token_ids_bl.shape[1], dtype=jnp.int32)[None, :]
        x_bld = x_bld + self.pos_embed(pos_ids)

        for i in range(self.num_layers):
            x_bld = self.blocks[i](x_bld)

        x_bld = self.out_ln(x_bld)
        logits_blv = self.embed.attend(x_bld)
        return logits_blv


class TransformerBlock(nnx.Module):
    def __init__(self, cfg: Config, layer_id: int, n_layers: int, *, rngs: nnx.Rngs):
        attn_name = _resolve_attention_name(cfg, layer_id=layer_id, n_layers=n_layers)
        if attn_name == "mha":
            self.attn = MHA(cfg=cfg, rngs=rngs)
        elif attn_name == "swa":
            self.attn = SWA(cfg=cfg, rngs=rngs)
        elif attn_name == "mla":
            self.attn = MLA(cfg=cfg, rngs=rngs)
        else:
            raise ValueError(f"Unsupported attention backend `{attn_name}`.")
        self.ffn = SwiGLU(cfg=cfg, rngs=rngs)
        self.ln1 = nnx.RMSNorm(num_features=cfg.dim, rngs=rngs)
        self.ln2 = nnx.RMSNorm(num_features=cfg.dim, rngs=rngs)

    def __call__(self, x_bld: jnp.ndarray) -> jnp.ndarray:
        a_bld = self.attn(self.ln1(x_bld))
        f_bld = self.ffn(self.ln2(x_bld))
        return x_bld + a_bld + f_bld


class CausalSelfAttention(nnx.Module):
    def __init__(self, cfg: Config, *, rngs: nnx.Rngs):
        if cfg.dim % cfg.num_heads != 0:
            raise ValueError("`dim` must be divisible by `num_heads`.")
        self.head_dim = cfg.dim // cfg.num_heads
        self.q = nnx.LinearGeneral(
            in_features=cfg.dim,
            out_features=(cfg.num_heads, self.head_dim),
            axis=-1,
            use_bias=False,
            rngs=rngs,
        )
        self.k = nnx.LinearGeneral(
            in_features=cfg.dim,
            out_features=(cfg.num_heads, self.head_dim),
            axis=-1,
            use_bias=False,
            rngs=rngs,
        )
        self.v = nnx.LinearGeneral(
            in_features=cfg.dim,
            out_features=(cfg.num_heads, self.head_dim),
            axis=-1,
            use_bias=False,
            rngs=rngs,
        )
        self.out = nnx.LinearGeneral(
            in_features=(cfg.num_heads, self.head_dim),
            out_features=cfg.dim,
            axis=(-2, -1),
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x_bld: jnp.ndarray) -> jnp.ndarray:
        q_blhd = self.q(x_bld)
        k_blhd = self.k(x_bld)
        v_blhd = self.v(x_bld)

        attn_bhll = jnp.einsum("bqhd,bkhd->bhqk", q_blhd, k_blhd)
        attn_bhll = attn_bhll / jnp.sqrt(self.head_dim)

        l = x_bld.shape[1]
        mask_11ll = _build_causal_mask(l, window=None)
        neg_inf = jnp.finfo(attn_bhll.dtype).min
        attn_bhll = jnp.where(mask_11ll, attn_bhll, neg_inf)
        attn_bhll = jax.nn.softmax(attn_bhll, axis=-1)

        y_blhd = jnp.einsum("bhqk,bkhd->bqhd", attn_bhll, v_blhd)
        return self.out(y_blhd)


class MHA(CausalSelfAttention):
    pass


class SWA(CausalSelfAttention):
    def __init__(self, cfg: Config, *, rngs: nnx.Rngs):
        super().__init__(cfg=cfg, rngs=rngs)
        if cfg.swa_window <= 0:
            raise ValueError("`swa_window` must be > 0.")
        self.window = cfg.swa_window

    def __call__(self, x_bld: jnp.ndarray) -> jnp.ndarray:
        q_blhd = self.q(x_bld)
        k_blhd = self.k(x_bld)
        v_blhd = self.v(x_bld)

        attn_bhll = jnp.einsum("bqhd,bkhd->bhqk", q_blhd, k_blhd)
        attn_bhll = attn_bhll / jnp.sqrt(self.head_dim)

        l = x_bld.shape[1]
        mask_11ll = _build_causal_mask(l, window=self.window)
        neg_inf = jnp.finfo(attn_bhll.dtype).min
        attn_bhll = jnp.where(mask_11ll, attn_bhll, neg_inf)
        attn_bhll = jax.nn.softmax(attn_bhll, axis=-1)

        y_blhd = jnp.einsum("bhqk,bkhd->bqhd", attn_bhll, v_blhd)
        return self.out(y_blhd)


class MLA(nnx.Module):
    def __init__(self, cfg: Config, *, rngs: nnx.Rngs):
        if cfg.dim % cfg.num_heads != 0:
            raise ValueError("`dim` must be divisible by `num_heads`.")
        if cfg.mla_latent_dim <= 0:
            raise ValueError("`mla_latent_dim` must be > 0.")

        self.head_dim = cfg.dim // cfg.num_heads
        self.latent_dim = cfg.mla_latent_dim
        self.q = nnx.LinearGeneral(
            in_features=cfg.dim,
            out_features=(cfg.num_heads, self.head_dim),
            axis=-1,
            use_bias=False,
            rngs=rngs,
        )
        self.kv_down = nnx.Linear(cfg.dim, self.latent_dim, use_bias=False, rngs=rngs)
        self.k_up = nnx.LinearGeneral(
            in_features=self.latent_dim,
            out_features=(cfg.num_heads, self.head_dim),
            axis=-1,
            use_bias=False,
            rngs=rngs,
        )
        self.v_up = nnx.LinearGeneral(
            in_features=self.latent_dim,
            out_features=(cfg.num_heads, self.head_dim),
            axis=-1,
            use_bias=False,
            rngs=rngs,
        )
        self.out = nnx.LinearGeneral(
            in_features=(cfg.num_heads, self.head_dim),
            out_features=cfg.dim,
            axis=(-2, -1),
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x_bld: jnp.ndarray) -> jnp.ndarray:
        q_blhd = self.q(x_bld)
        latent_bll = self.kv_down(x_bld)
        k_blhd = self.k_up(latent_bll)
        v_blhd = self.v_up(latent_bll)

        attn_bhll = jnp.einsum("bqhd,bkhd->bhqk", q_blhd, k_blhd)
        attn_bhll = attn_bhll / jnp.sqrt(self.head_dim)

        l = x_bld.shape[1]
        mask_11ll = _build_causal_mask(l, window=None)
        neg_inf = jnp.finfo(attn_bhll.dtype).min
        attn_bhll = jnp.where(mask_11ll, attn_bhll, neg_inf)
        attn_bhll = jax.nn.softmax(attn_bhll, axis=-1)

        y_blhd = jnp.einsum("bhqk,bkhd->bqhd", attn_bhll, v_blhd)
        return self.out(y_blhd)


class SwiGLU(nnx.Module):
    def __init__(self, cfg: Config, *, rngs: nnx.Rngs):
        self.gate = nnx.Linear(cfg.dim, cfg.ffn_dim, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(cfg.dim, cfg.ffn_dim, use_bias=False, rngs=rngs)
        self.out = nnx.Linear(cfg.ffn_dim, cfg.dim, use_bias=False, rngs=rngs)

    def __call__(self, x_bld: jnp.ndarray) -> jnp.ndarray:
        gate = nnx.swish(self.gate(x_bld))
        value = self.value(x_bld)
        return self.out(gate * value)


def _resolve_attention_name(cfg: Config, *, layer_id: int, n_layers: int) -> str:
    every = max(1, int(cfg.attn_global_every))
    if every == 1:
        return cfg.attn.lower()
    is_last = layer_id == (n_layers - 1)
    is_global = ((layer_id + 1) % every == 0) or is_last
    return (cfg.attn if is_global else cfg.attn_local).lower()


def _build_causal_mask(ctx_len: int, window: int | None) -> jnp.ndarray:
    q = jnp.arange(ctx_len)[:, None]
    k = jnp.arange(ctx_len)[None, :]
    mask = k <= q
    if window is not None:
        mask = mask & (k >= (q - window + 1))
    return mask[None, None, :, :]
