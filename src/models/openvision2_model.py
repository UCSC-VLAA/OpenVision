# Copyright 2025 Yanqing Liu.
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

"""Transformer-based multi-modal model for image and text processing.

This module defines a flexible architecture that combines a Vision Transformer (ViT)
for image encoding with a generic text decoder for tasks like image captioning.
It is designed to be modular, allowing different model configurations to be
dynamically imported and used.

The primary components are:
- An image encoder (e.g., ViT) that processes images into a sequence of embeddings.
- A text decoder (e.g., a Transformer decoder) that uses the image embeddings
  as context to generate text.

The file also includes a utility function for loading pre-trained weights into
the model's components.

This code is based on materials from the Big Vision project:
https://github.com/google-research/big_vision
"""

# Standard library imports
import importlib
from typing import Any, Dict, Optional, Union

# Third-party imports
import flax.linen as nn
import jax
import jax.numpy as jnp

# Type alias for configuration dictionaries
ConfigDict = Dict[str, Any]


class Model(nn.Module):
    """A multi-modal model combining an image encoder and a text decoder.

    This class serves as a container that orchestrates the forward pass through
    an image model (ViT) and a text decoder. It is designed for tasks such as
    image captioning where image features are used to condition text generation.

    Attributes:
        image: Configuration dictionary for the image encoder model.
        text_decoder_config: Configuration dictionary for the text decoder model.
        image_model_name: The name of the module containing the image model's
            `Model` class (e.g., "vit").
        text_decoder_name: The name of the module containing the text decoder's
            `Model` class.
        quick_gelu: If True, uses a faster GELU approximation.
        cast_dtype: Optional dtype to cast inputs to, for mixed-precision training.
        pad_id: The token ID used for padding in text sequences.
        mesh: The JAX device mesh for model parallelism.
        keep_ratio: The fraction of image patch embeddings to keep and feed as
            context to the text decoder. This implements a form of token dropping
            for efficiency.
    """
    image: Optional[ConfigDict] = None
    text_decoder_config: Optional[ConfigDict] = None
    image_model_name: str = "vit"
    text_decoder_name: Optional[str] = None
    quick_gelu: bool = False
    cast_dtype: Optional[jnp.dtype] = None
    pad_id: int = 0
    mesh: Any = None
    keep_ratio: float = 0.25

    @nn.compact
    def __call__(self,
                 image_batch_pixels: Optional[jnp.ndarray],
                 text_tokens: Optional[jnp.ndarray] = None,
                 train: bool = False,
                 **kw) -> Dict[str, jnp.ndarray]:
        """Performs the forward pass for image encoding and text decoding.

        Args:
            image_batch_pixels: A batch of raw input images, typically with shape
                `[batch_size, height, width, channels]`.
            text_tokens: A batch of tokenized text sequences for the decoder,
                typically with shape `[batch_size, sequence_length]`.
            train: A boolean indicating if the model is in training mode.
            **kw: Additional keyword arguments passed to the sub-modules.

        Returns:
            A dictionary containing the output logits from the text decoder
            under the key "logits".
        """
        out_dict = {}
        image_embs_for_captioning = None

        # --- 1. Image Encoder (ViT) ---
        if image_batch_pixels is not None:
            # Dynamically import and instantiate the image encoder model.
            vit_module = importlib.import_module(f"models.{self.image_model_name}")
            vit_encoder_module = vit_module.Model(
                **(self.image or {}), name="img_encoder", mesh=self.mesh, **kw
            )

            # Get image patch embeddings. Pooled output is not used here.
            _vit_pooled_output, image_embs = vit_encoder_module(
                image_batch_pixels, train=train, **kw
            )

            # During inference or training, randomly sample a subset of patch
            # embeddings to pass to the decoder. This is a form of attention
            # or data reduction.
            if not self.is_initializing():
                B, N, D = image_embs.shape  # [Batch, Num_Tokens, Dim]
                rng = self.make_rng("dropout")
                num_keep = int(N * self.keep_ratio)

                # Define a function to sample indices for a single example.
                def mask_fn(rng_i, x_i):
                    kept_idx = jax.random.choice(rng_i, N, shape=(num_keep,), replace=False)
                    return jnp.take(x_i, kept_idx, axis=0)

                # Efficiently apply the sampling to each example in the batch.
                rngs = jax.random.split(rng, B)
                image_embs_for_captioning = jax.vmap(mask_fn)(rngs, image_embs)
            else:
                # During initialization, image_embs might not be available yet.
                image_embs_for_captioning = None

        # --- 2. Text Decoder (for Captioning) ---
        if self.text_decoder_name is not None and text_tokens is not None:
            if image_embs_for_captioning is None and not self.is_initializing():
                raise ValueError(
                    "Image embeddings are required for the text decoder but were not generated."
                )

            # Dynamically import and instantiate the text decoder model.
            text_decoder_module = importlib.import_module(f"models.{self.text_decoder_name}").Model(
                **(self.text_decoder_config or {}), name="txt_decoder", **kw
            )

            current_image_condition = image_embs_for_captioning
            # During model initialization, create dummy image embeddings to ensure
            # the text decoder is initialized with the correct input shapes.
            if self.is_initializing() and current_image_condition is None:
                B_txt, _ = text_tokens.shape
                img_cond_dim = self.image.get('width', 1024)
                patch_size = self.image.get('patch_size', [14, 14])
                image_res = self.image.get('res', 224)
                num_dummy_img_tokens = (image_res // patch_size[0]) ** 2
                current_image_condition = jnp.zeros(
                    (B_txt, num_dummy_img_tokens, img_cond_dim), dtype=jnp.float32
                )

            # Generate logits for the next token prediction.
            logits_for_text_pred = text_decoder_module(
                text_input=text_tokens,
                context=current_image_condition,
                train=train, **kw
            )
            out_dict["logits"] = logits_for_text_pred

        return out_dict


def load(init_params: Dict,
         init_files: Union[str, Dict[str, str]],
         model_cfg: ConfigDict,
         img_load_kw: Optional[Dict] = None,
         txt_load_kw: Optional[Dict] = None) -> Dict:
    """Loads pre-trained weights for the model's components.

    This function handles loading weights for the image encoder and text decoder
    from separate checkpoint files.

    Args:
        init_params: A dictionary of initial model parameters (e.g., from an
            initialization function). This structure is used as a template.
        init_files: A dictionary mapping component names (e.g., "img_encoder",
            "txt_decoder") to their checkpoint file paths. Can also be a single
            string, assumed to be the path for the image encoder.
        model_cfg: The model's configuration object, which contains names and
            configs for the sub-modules.
        img_load_kw: Optional keyword arguments for the image model's load function.
        txt_load_kw: Optional keyword arguments for the text model's load function.

    Returns:
        A dictionary of parameters with the pre-trained weights loaded.
    """
    img_load_kw = img_load_kw or {}
    txt_load_kw = txt_load_kw or {}
    
    # Standardize `init_files` to be a dictionary for consistent processing.
    if isinstance(init_files, str):
        init_files_dict = {"img_encoder": init_files}
    else:
        init_files_dict = {**init_files}

    restored_params = {**init_params}

    # --- Load Image Encoder (ViT) ---
    # Allow flexible keys like "img" or "img_encoder".
    img_encoder_path = init_files_dict.pop("img_encoder", init_files_dict.pop("img", None))
    if img_encoder_path and "img_encoder" in init_params:
        if hasattr(model_cfg, 'image_model_name') and hasattr(model_cfg, 'image'):
            # Dynamically import the corresponding load function.
            load_fn = importlib.import_module(f"models.{model_cfg.image_model_name}").load
            restored_params["img_encoder"] = load_fn(
                init_params["img_encoder"], img_encoder_path, model_cfg.image, **img_load_kw
            )
        else:
            print("Warning: `image_model_name` or `image` config not found. Skipping image encoder loading.")

    # --- Load Text Decoder ---
    txt_decoder_path = init_files_dict.pop("txt_decoder", None)
    if txt_decoder_path and "txt_decoder" in init_params:
        if hasattr(model_cfg, 'text_decoder_name') and model_cfg.text_decoder_name:
            load_fn = importlib.import_module(f"models.{model_cfg.text_decoder_name}").load
            restored_params["txt_decoder"] = load_fn(
                init_params["txt_decoder"], txt_decoder_path, model_cfg.text_decoder_config, **txt_load_kw
            )
        else:
            print("Warning: `text_decoder_name` not specified. Skipping text decoder loading.")

    # Report any unused checkpoint file paths to the user.
    if init_files_dict:
        print(f"Warning: Unused keys in `init_files`: {list(init_files_dict.keys())}")

    return restored_params