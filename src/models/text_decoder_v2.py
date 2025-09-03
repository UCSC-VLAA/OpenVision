# #Copyright @2024 Yanqing Liu
#
# # This code is based on materials from the Big Vision [https://github.com/google-research/big_vision].
# # Thanks to Big Vision  for their contributions to the field of computer vision and for their open-source contributions to this project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines a flexible image-to-text autoregressive Transformer model.

This module provides the building blocks for a decoder-style Transformer that
can be conditioned on image embeddings to generate text. It features a unique
architecture that alternates between self-attention and cross-attention layers.

Key features include:
- A `MultiHeadDotProductAttention` layer with optional support for TPU Flash
  Attention for enhanced performance.
- A `CrossAttnEncoder` that interleaves self-attention on text with
  cross-attention to image features.
- Support for different fusion strategies ('concat' and 'cross_attn').
- Model parallelism support via Flax's logical partitioning APIs.
- Gradient checkpointing (re-materialization) to save memory during training.
"""

# Standard library imports
import functools
from typing import (Any, Callable, Optional, Sequence, Tuple, Union)

# Third-party imports
from absl import logging
from einops import rearrange, repeat
import flax.linen as nn
from flax.linen.module import compact
from flax.linen.module import merge_param
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu import flash_attention as tpu_flash_attention
from jax.experimental.shard_map import shard_map

# Local application/library specific imports
from src.helpers import utils
from src.models import common
from src.models.common import DropPath
from src.models.text_transformer import Encoder, MlpBlock, Encoder1DBlock

# Type Aliases for clarity
Array = Any
Dtype = Any


class MultiHeadDotProductAttention(nn.MultiHeadDotProductAttention):
    """Multi-head dot-product attention with optional TPU Flash Attention.

    This class extends Flax's default MHA to include support for a highly
    optimized TPU flash attention kernel, custom kernel initializers, and
    integration with model parallelism through logical partitioning.

    Attributes:
        attn_kernel_init: Initializer for the attention weights (query, key, value).
        proj_kernel_init: Initializer for the output projection weights.
        use_flash_attn: If True, uses the Pallas TPU flash attention implementation.
        dtype: The dtype of the computation (e.g., bfloat16).
        param_dtype: The dtype of the model parameters (e.g., float32).
        mesh: The JAX device mesh for model parallelism.
    """
    attn_kernel_init: Callable = nn.initializers.normal(stddev=0.01)
    proj_kernel_init: Callable = nn.initializers.normal(stddev=0.01)
    use_flash_attn: bool = False
    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Dtype = jnp.float32
    mesh: Optional[Any] = None

    @compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array,
                 mask: Optional[Array] = None,
                 deterministic: Optional[bool] = None) -> Array:
        """Applies multi-head dot product attention.

        Args:
            inputs_q: Input queries of shape `[B, L_q, D]`.
            inputs_kv: Keys/values of shape `[B, L_kv, D]`.
            mask: Attention mask.
            deterministic: If true, disables dropout.

        Returns:
            The output of the attention mechanism, with shape `[B, L_q, D]`.
        """
        # Apply logical constraints for model parallelism
        inputs_q = nn.with_logical_constraint(
            inputs_q, ("activation_batch", "activation_length", "activation_embed"))
        inputs_kv = nn.with_logical_constraint(
            inputs_kv, ("activation_batch", "activation_length", "activation_embed"))

        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        if qkv_features % self.num_heads != 0:
            raise ValueError(
                'Memory dimension must be divisible by number of heads.')
        head_dim = qkv_features // self.num_heads

        dense = functools.partial(
            nn.DenseGeneral,
            axis=-1,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            features=(self.num_heads, head_dim),
            kernel_init=nn.with_logical_partitioning(
                self.attn_kernel_init, ("embed", "heads")),
            bias_init=nn.with_logical_partitioning(self.bias_init, (None,)),
            use_bias=self.use_bias,
            precision=self.precision)

        # Project inputs to query, key, value
        query, key, value = (dense(name='query')(inputs_q),
                             dense(name='key')(inputs_kv),
                             dense(name='value')(inputs_kv))
        
        # Determine if dropout should be applied
        dropout_rng = None
        m_deterministic = merge_param('deterministic', self.deterministic, deterministic)
        if self.dropout_rate > 0. and not m_deterministic:
            dropout_rng = self.make_rng('dropout')

        # Cast to computation dtype
        query = query.astype(self.dtype)
        key = key.astype(self.dtype)
        value = value.astype(self.dtype)

        if not self.use_flash_attn:
            # Use standard Flax attention implementation
            x = self.attention_fn(
                query, key, value,
                mask=mask,
                dropout_rng=dropout_rng,
                dropout_rate=self.dropout_rate,
                broadcast_dropout=self.broadcast_dropout,
                deterministic=m_deterministic,
                dtype=self.dtype,
                precision=self.precision)
        else:
            # Use optimized TPU Flash Attention
            x = self._tpu_flash_attention(query, key, value)
        
        # Output projection
        x = nn.DenseGeneral(
            features=features,
            axis=(-2, -1),
            kernel_init=nn.with_logical_partitioning(
                self.proj_kernel_init, ("heads", "embed")),
            bias_init=nn.with_logical_partitioning(self.bias_init, (None,)),
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            precision=self.precision,
            name='out')(x)

        return x

    def _tpu_flash_attention(self, query: Array, key: Array, value: Array) -> Array:
        """Executes the TPU Flash Attention kernel via shard_map."""
        if not self.mesh:
            raise ValueError(
                'A device mesh must be provided to use flash attention.')

        # Transpose to [B, H, L, D_h] format expected by kernel
        query = jnp.transpose(query, axes=(0, 2, 1, 3))
        key = jnp.transpose(key, axes=(0, 2, 1, 3))
        value = jnp.transpose(value, axes=(0, 2, 1, 3))
        
        # Define how tensors are sharded across the device mesh
        # Shards batch dim across 'data' and 'fsdp' mesh axes
        axis_names = jax.sharding.PartitionSpec(("data", "fsdp"), None, None, None)

        @functools.partial(
            shard_map,
            mesh=self.mesh,
            in_specs=(axis_names, axis_names, axis_names),
            out_specs=axis_names,
            check_rep=False)
        def wrap_flash_attention(q, k, v):
            return tpu_flash_attention.flash_attention(
                q, k, v, causal=False, sm_scale=1.0 / np.sqrt(q.shape[-1]))

        # Batch dimension must be divisible by the product of data and fsdp axes sizes
        devices_in_data_fsdp = self.mesh.shape["data"] * self.mesh.shape["fsdp"]
        if query.shape[0] % devices_in_data_fsdp != 0:
            raise ValueError(
                "Batch dimension must be shardable across data and fsdp mesh axes.")

        x = wrap_flash_attention(query, key, value)
        # Transpose back to [B, L, H, D_h]
        x = jnp.transpose(x, axes=(0, 2, 1, 3))
        return x


class CrossAttnEncoder1DBlock(nn.Module):
    """A Transformer block with cross-attention followed by an MLP.

    This block takes two inputs: `x` (typically text embeddings) which acts
    as the query, and `u` (typically image embeddings) which acts as the
    key and value for the cross-attention layer.

    Attributes:
        mlp_dim: The hidden dimension of the MLP block.
        num_heads: The number of attention heads.
        dropout: Dropout rate for attention and MLP outputs.
        drop_path: Stochastic depth drop rate.
        depth: The total number of layers in the parent encoder, used for weight init scaling.
        use_flash_attn: Whether to use TPU Flash Attention.
        dtype: The computation data type.
        param_dtype: The parameter data type.
        mesh: The JAX device mesh for model parallelism.
    """
    mlp_dim: Optional[int] = None
    num_heads: int = 12
    dropout: float = 0.0
    drop_path: float = 0.0
    depth: int = 12
    use_flash_attn: bool = False
    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Dtype = jnp.float32
    mesh: Optional[Any] = None

    @nn.compact
    def __call__(self, x: Array, u: Array, deterministic: bool = True) -> Tuple[Array, dict]:
        width = x.shape[-1]
        # Scaled weight initialization based on model depth
        init_std = {
            'proj': (width ** -0.5) * ((2 * self.depth) ** -0.5),
            'attn': width ** -0.5,
            'fc': (2 * width) ** -0.5
        }
        out = {}

        # Cross-Attention part
        y = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="ln_x")(x)
        v = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="ln_u")(u)
        
        y = out["cross_attn"] = MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            attn_kernel_init=nn.initializers.normal(stddev=init_std['attn']),
            proj_kernel_init=nn.initializers.normal(stddev=init_std['proj']),
            bias_init=nn.initializers.zeros,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            deterministic=deterministic,
            use_flash_attn=self.use_flash_attn,
            mesh=self.mesh,
        )(inputs_q=y, inputs_kv=v)
        
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        y = DropPath(dropout_prob=self.drop_path)(y, deterministic)
        x = x + y

        # MLP part
        y = nn.LayerNorm(dtype=self.dtype, param_dtype=self.param_dtype, name="ln_mlp")(x)
        y = out["mlp"] = MlpBlock(
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
            fc_init=nn.initializers.normal(stddev=init_std['fc']),
            proj_init=nn.initializers.normal(stddev=init_std['proj']),
            param_dtype=self.param_dtype,
            dtype=self.dtype,
        )(y, deterministic)
        
        y = nn.Dropout(rate=self.dropout)(y, deterministic)
        y = DropPath(dropout_prob=self.drop_path)(y, deterministic)
        x = x + y
        
        return x, out


class CrossAttnEncoder(nn.Module):
    """A Transformer Encoder with interleaved self-attention and cross-attention.

    This architecture processes a primary sequence `x` (e.g., text) by alternating
    between self-attention blocks (to process `x` itself) and cross-attention
    blocks (to infuse information from a context sequence `u`, e.g., an image).

    Attributes:
        depth: The total number of layers. Each "layer" consists of one
               self-attention block and one cross-attention block.
        remat_policy: Gradient checkpointing policy ('none', 'minimal', etc.).
    """
    depth: int
    mlp_dim: Optional[int] = None
    num_heads: int = 12
    dropout: float = 0.0
    drop_path: float = 0.0
    remat_policy: str = "none"
    casual_mask: bool = False
    use_flash_attn: bool = False
    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Dtype = jnp.float32
    mesh: Optional[Any] = None

    @nn.compact
    def __call__(self, x: Array, u: Array, deterministic: bool = True) -> Tuple[Array, dict]:
        out = {}
        # Linearly increasing drop path rate for stochastic depth
        dpr = [rate.item() for rate in np.linspace(0, self.drop_path, self.depth)]
        
        # Configure gradient checkpointing (remat) if specified
        if self.remat_policy not in (None, "none"):
            policy = (jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
                      if self.remat_policy == "minimal" else None)
            logging.info(f"Applying activation checkpointing: {self.remat_policy}")
            SelfAttnBlock = nn.remat(
                Encoder1DBlock, prevent_cse=True, policy=policy, static_argnums=(1,))
            CrossAttnBlock = nn.remat(
                CrossAttnEncoder1DBlock, prevent_cse=True, policy=policy, static_argnums=(2,))
        else:
            SelfAttnBlock = Encoder1DBlock
            CrossAttnBlock = CrossAttnEncoder1DBlock
            
        for i in range(self.depth):
            # 1. Self-attention on text embeddings `x`
            x, out[f"self_attn_block_{i:02d}"] = SelfAttnBlock(
                name=f"self_attn_block_{i}",
                mlp_dim=self.mlp_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                dropout=self.dropout,
                drop_path=dpr[i],
                casual_mask=self.casual_mask,
                use_flash_attn=self.use_flash_attn,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                mesh=self.mesh
            )(x, deterministic)
            
            # 2. Cross-attention from text `x` to image context `u`
            x, out[f"cross_attn_block_{i:02d}"] = CrossAttnBlock(
                name=f"cross_attn_block_{i}",
                mlp_dim=self.mlp_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                dropout=self.dropout,
                drop_path=dpr[i],
                use_flash_attn=self.use_flash_attn,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                mesh=self.mesh
            )(x, u, deterministic)
            
        return x, out


class _Model(nn.Module):
    """Core implementation of the image-text autoregressive Transformer.

    Attributes:
        fusion_style: Method to combine image and text features.
            'concat': Concatenate image and text embeddings and process with a
                      standard decoder.
            'cross_attn': Use the interleaved self/cross-attention encoder.
        vocab_size: The size of the text vocabulary.
    """
    width: int = 512
    depth: int = 12
    mlp_dim: Optional[int] = None
    num_heads: int = 12
    dropout: float = 0.0
    drop_path: float = 0.0
    remat_policy: str = 'none'
    fusion_style: str = 'cross_attn'
    casual_mask: bool = True
    use_flash_attn: bool = False
    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Dtype = jnp.float32
    mesh: Optional[Any] = None
    vocab_size: int = 32000

    @nn.compact
    def __call__(self, text_input: Array, context: Array, *, train: bool = False) -> Array:
        out = {}
        # The model predicts the next token, so we use tokens up to the second to last.
        token_ids_in = text_input[:, :-1]

        embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.width,
            dtype=jnp.float32,  # Use float32 for stability
            param_dtype=self.param_dtype,
            embedding_init=nn.with_logical_partitioning(
                nn.initializers.normal(stddev=0.02), ('vocab', 'embed')))
        text_embeds = embedding(token_ids_in.astype("int32"))

        # Project image features to the same dimension as text embeddings.
        _, _, img_dim = context.shape
        image_projection_layer = nn.Dense(
            self.width,
            name="image_projection_layer",
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=img_dim ** -0.5))
        image_embeds = image_projection_layer(context)

        # --- FUSION OF IMAGE AND TEXT FEATURES ---
        if self.fusion_style == 'concat':
            img_len = image_embeds.shape[1]
            image_text_embeds = jnp.concatenate((image_embeds, text_embeds), axis=1)

            # Use a standard causal decoder over the concatenated sequence.
            decoder_blocks = Encoder(
                name="Transformer", **self.get_transformer_kwargs())
            x, _ = decoder_blocks(image_text_embeds, deterministic=not train)
            
            # Discard the output corresponding to image tokens.
            x = x[:, img_len:]

        elif self.fusion_style == 'cross_attn':
            if self.depth % 2 != 0:
                raise ValueError("Depth must be even for cross_attn fusion style.")
            
            # Use the interleaved self-attention/cross-attention encoder.
            decoder_blocks = CrossAttnEncoder(
                name="Transformer", depth=self.depth // 2,
                **self.get_transformer_kwargs())
            x, _ = decoder_blocks(text_embeds, image_embeds, deterministic=not train)
        
        else:
            raise ValueError(f"Unknown fusion_style: '{self.fusion_style}'")

        # Final LayerNorm and output projection head.
        x = nn.LayerNorm(name="decoder_norm")(x)
        logits = nn.Dense(
            self.vocab_size,
            name="head",
            use_bias=False,
            dtype=jnp.float32,  # Use float32 for stable logits
            param_dtype=self.param_dtype,
            kernel_init=nn.with_logical_partitioning(
                nn.initializers.normal(stddev=self.width ** -0.5), ("embed", "vocab"))
        )(x)

        return logits

    def get_transformer_kwargs(self) -> dict:
        """Helper to gather shared Transformer arguments."""
        return {
            "mlp_dim": self.mlp_dim,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "drop_path": self.drop_path,
            "remat_policy": self.remat_policy,
            "casual_mask": self.casual_mask,
            "use_flash_attn": self.use_flash_attn,
            "dtype": self.dtype,
            "param_dtype": self.param_dtype,
            "mesh": self.mesh,
        }


def Model(variant: Optional[str] = None, **kw) -> _Model:
    """Factory function to create the autoregressive model with a variant.

    Args:
        variant: A string shorthand for a model size (e.g., 'B', 'L').
        **kw: Additional keyword arguments to override variant defaults.

    Returns:
        An instance of the _Model class.
    """
    variant_args = decode_variant(variant)
    return _Model(**{**variant_args, **kw})


def decode_variant(variant: Optional[str]) -> dict:
    """Converts a model variant string into a dictionary of hyperparameters."""
    if not variant:
        return {}

    # Hyperparameters from text transformer models, often similar to ViT sizes.
    return {
        "width": {"Ti": 192, "S": 384, "M": 512, "B": 512, "L": 768, "H": 1024, "G": 1664}[variant],
        "depth": {"Ti": 12, "S": 12, "M": 12, "B": 12, "L": 12, "H": 24, "G": 48}[variant],
        "mlp_dim": {"Ti": 768, "S": 1536, "M": 2048, "B": 2048, "L": 3072, "H": 4096, "G": 8192}[variant],
        "num_heads": {"Ti": 3, "S": 6, "M": 8, "B": 8, "L": 12, "H": 16, "G": 16}[variant],
    }


def load(init_params: dict, init_file: str, model_cfg: dict, dont_load: Sequence[str] = ()) -> dict:
    """Loads pre-trained model weights from a checkpoint file.

    Args:
        init_params: A PyTree of randomly initialized model parameters.
        init_file: Path to the checkpoint file to load.
        model_cfg: The model configuration dictionary (unused in this function).
        dont_load: A sequence of parameter names (or prefixes) to exclude from loading.

    Returns:
        A PyTree of parameters with loaded weights.
    """
    del model_cfg
    restored_params = utils.load_params(filepath=init_file)

    # Merge restored params into the initialized structure, allowing for fine-tuning
    # where some parameters (like the output head) might be re-initialized.
    restored_params = common.merge_params(
        restored_params, init_params, dont_load=dont_load)

    # Ensure dtypes are correctly restored (e.g., from float32 to bfloat16).
    restored_params = jax.tree_util.tree_map(
        utils.recover_dtype, restored_params)
        
    return restored_params