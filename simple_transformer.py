from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from config import Config


class Transformer(nnx.Module):
    def __init__(self, cfg: Config, mesh: jax.sharding.Mesh, *, rngs: nnx.Rngs):
        init_fn = nnx.initializers.lecun_normal()
        scale_fn = nnx.initializers.ones
        self.mesh = mesh
        self.embed = nnx.Embed(
            cfg.vocab_size,
            cfg.dim,
            embedding_init=partial(init_fn, out_sharding=NamedSharding(mesh, P())),
            rngs=rngs,
        )
        self.pos_embed = nnx.Embed(cfg.ctx_len, cfg.dim,
            embedding_init=partial(init_fn, out_sharding=NamedSharding(mesh, P())),
                                   rngs=rngs)
        self.out_ln = nnx.RMSNorm(
            num_features=cfg.dim,
            scale_init=partial(scale_fn, out_sharding=NamedSharding(mesh, P())),
            rngs=rngs,
        )

    def __call__(self, token_ids_bl: jnp.ndarray) -> jnp.ndarray:
        x_bld = self.embed(token_ids_bl, out_sharding=NamedSharding(self.mesh, P("batch",)))
        pos_ids = jnp.arange(token_ids_bl.shape[1], dtype=jnp.int32)[None, :]
        x_bld = x_bld + self.pos_embed(pos_ids)
        x_bld = self.out_ln(x_bld)
        return self.embed.attend(x_bld)
