# Copyright 2022 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long
r"""OpenVision 2 training config (JAX/TPU).

This is a sanitized, open-source friendly template derived from an internal
config. It keeps the same training and eval structure while replacing
private paths and credentials with placeholders.

Typical usage:
  python3 -m main_clip \
    --config=configs/openvision2.py:\
res=84,img=L/16,txt_decoder_name=L,base_lr=8e-6,keep_ratio=0.35,\
batch_factor=2,data_parallelism=256,fsdp_parallelism=1,tensor_parallelism=1,\
imagenet_epoch=10000,vitual_warmup_epoch=40

"""

import functools
import jax.numpy  # noqa: F401

import configs.common as bvcc
import configs.clip_common as common  # noqa: F401
from ml_collections import ConfigDict


def get_config(arg=None):
  """Builds the base configuration."""
  arg = bvcc.parse_arg(
      arg,
      # Core training args
      res=112,
      batch_factor=2.0,
      base_lr=8e-6,
      imagenet_epoch=2000,
      vitual_warmup_epoch=20,
      runlocal=False,
      # Text
      token_len=128,
      txt='bert_base',
      txt_key='llava_llama3_condition_True_dense_weighted_topk',
      use_openclip_tokenizer=False,
      # Image masking
      keep_ratio=0.35,
      # Models
      img='L/16',
      remat='full',
      img_head=True,
      load_pretrain=False,
      init='',
      txt_decoder_name='L',
      vocab_size=32000,
      # Parallelism
      data_parallelism=128,
      fsdp_parallelism=2,
      tensor_parallelism=1,
  )

  config = ConfigDict()

  #####################################
  #            sharding               #
  #####################################
  config.sharding = dict()
  config.sharding.meshshape = dict(
      data_parallelism=arg.data_parallelism,
      fsdp_parallelism=arg.fsdp_parallelism,
      tensor_parallelism=arg.tensor_parallelism,
  )
  config.sharding.mesh_axes = ['data', 'fsdp', 'tensor']
  config.sharding.data_sharding = [['data', 'fsdp', 'tensor']]

  # Logical axis rules. Adjust if your model layers differ.
  config.sharding.logical_axis_rules = [
      ['activation_batch', ['data', 'fsdp']],
      ['activation_heads', ['tensor']],
      ['activation_length', []],
      ['activation_embed', ['tensor']],
      ['activation_mlp', ['tensor']],
      ['activation_kv', ['tensor']],
      ['activation_vocab', ['tensor']],
      ['mlp', 'tensor'],
      ['vocab', 'tensor'],
      ['embed', 'fsdp'],
      ['norm', 'tensor'],
      ['heads', 'tensor'],
      ['kv', []],
  ]

  #####################################
  #              W&B                  #
  #####################################
  # Do NOT put tokens here. Use WANDB_API_KEY env var when launching.
  config.wandb = dict(
      log_wandb=True,
      wandb_offline=False,
      resume=False,
      debug_data=False,
      project='openvision2',  # placeholder
      entity='your-wandb-entity',  # placeholder
      experiment=f'OV2_{arg.img}_gbs_{int(1024*16*arg.batch_factor)}_res_{arg.res}_tok_{arg.token_len}_lr{arg.base_lr}',
  )

  #####################################
  #        bookkeeping / logging      #
  #####################################
  config.save_ckpt = True
  config.keep_ckpt = 100000000
  config.ckpt_steps = 1000
  config.log_training_steps = 50

  #####################################
  #               input               #
  #####################################
  config.input = {}
  # Replace with your dataset name and place. Example uses a generic "datacomp1b".
  config.input.data = dict(
      name='datacomp1b',                 # placeholder dataset name in your input pipeline
      split='full',
      data_dir='gs://your-bucket/datacomp1b',  # placeholder GCS path
  )
  config.input.cach_raw = True
  config.input.shuffle_buffer_size = 250_000 if not arg.runlocal else 50
  config.input.txt_token_length = arg.token_len
  config.init_shapes = [(256, arg.res, arg.res, 3), (256, arg.token_len,)]
  config.init_types = ['float32', 'int32']

  # Tokenizer selection
  # Provide a local vocab file in your repo, or point to GCS/HTTP as needed.
  vocab_path = 'assets/bert_base_vocab_bos_eos.txt'
  if arg.use_openclip_tokenizer:
    text_pp = '|flatten|copy("texts", "labels")|keep("image", "labels")'
  else:
    tokenizer = (
        f'my_bert_tokenize_v2(max_len={arg.token_len}, '
        f'vocab_path="{vocab_path}", add_bos=True, add_eos=True, key="{arg.txt_key}")'
    )
    # get_autoreg_label should create "autoreg_labels" and "cap_loss_mask"
    text_pp = (
        f'|flatten|{tokenizer}|get_autoreg_label(pad_token=0)'
        '|keep("image", "labels", "autoreg_labels", "cap_loss_mask")'
    )

  # Image preprocessing
  if getattr(arg, 'color_jitter', True):
    input_pp = (
        f'inception_crop(inkey="jpg", size={arg.res}, area_min=40, '
        f'method="bilinear", antialias=True)|simclr_jitter_gray(jitter_strength=0.4)'
    )
  else:
    input_pp = (
        f'inception_crop(inkey="jpg", size={arg.res}, area_min=40, '
        f'method="bilinear", antialias=True)'
    )

  config.input.pp = input_pp + text_pp
  config.pp_modules = ['ops_general', 'ops_image', 'ops_text', 'bert_ops']

  #####################################
  #               model               #
  #####################################
  config.model_name = 'openvision2_model'
  config.model_load = {}

  config.model = ConfigDict()
  config.model.image_model_name = 'vit'
  config.model.text_decoder_name = 'text_decoder_v2'
  config.model.text_decoder_config = ConfigDict({
      'variant': arg.txt_decoder_name,
      'num_classes': arg.vocab_size,
      'remat_policy': 'none',
      'fusion_style': 'concat',
      'casual_mask': True,  # keep as in your original
  })
  config.model.image = ConfigDict({
      'variant': arg.img,
      'posemb': 'sincos2d',
      'remat_policy': arg.remat,
      'use_flash_attn': False,
      'emb_head_bias': False,
      'head_zeroinit': False,
      'dtype': 'float32',
      'param_dtype': 'float32',
      'use_dense_general': False,
      'pool_type': 'gap',
      'output_tokens': True
  })
  variant_key = config.model.image.variant.split('/')[0]
  config.model.image.width = {
      'mu': 32, 'Ti': 192, 'S': 384, 'M': 512, 'B': 768,
      'L': 1024, 'So400m': 1152, 'H': 1280,
      'g': 1408, 'g-opt': 1536, 'G': 1664, 'G-opt': 1536, 'e': 1792,
  }[variant_key]

  config.model.keep_ratio = arg.keep_ratio
  config.optax_name = 'scale_by_adam'
  config.ft_from = ''
  config.masked_init = ''

  #####################################
  #          batch and steps          #
  #####################################
  config.input.batch_size = int(1024 * 16 * arg.batch_factor)
  batch_size = config.input.batch_size

  imagenet_samples = 1_281_167
  vitual_imagenet_epoch = arg.imagenet_epoch
  vitual_warmup_epoch = arg.vitual_warmup_epoch
  total_seen_samples = imagenet_samples * vitual_imagenet_epoch
  total_warmup_samples = imagenet_samples * vitual_warmup_epoch

  config.total_steps = int(total_seen_samples // batch_size) if not arg.runlocal else 1

  # Learning rate scales with global batch. Reference value for gbs=16k
  config.lr = arg.base_lr * 64 * arg.batch_factor
  config.wd = 0.2
  warmup_steps = int(total_warmup_samples // batch_size)
  config.schedule = [
      ('.*', dict(
          decay_type='cosine',
          warmup_steps=warmup_steps,
          min_lr=0.0,
          max_lr=arg.base_lr * 64 * arg.batch_factor,
      )),
  ]
  config.optax = dict(mu_dtype='bfloat16', b1=0.9, b2=0.95)

  #####################################
  #             training              #
  #####################################
  config.coca_caption_loss_weight = 2
  config.loss_use_global_batch = True
  config.local_loss = True
  config.cpu_unit8 = True  # keep flag name as in your codebase

  # Mixup/Cutmix off by default here, keep knobs available
  config.input.use_mixup = False
  config.input.mixup = dict(p=0.8, fold_in=None)
  config.input.cutmix = dict(alpha=1.0, beta=1.0)
  config.input.switch_prob = 0.5

  #####################################
  #               eval                #
  #####################################
  config.eval_only = False
  eval_common = dict(
      type='proj.image_text.contrastive',
      use_global_batch=config.loss_use_global_batch,
      log_steps=int(2000 // arg.batch_factor),
  )
  config.evals = {}

  sub = '[:4]' if arg.runlocal else ''

  def tokenizer_eval(inkey):
    return (
        f'my_eval_bert_tokenize(inkey="{inkey}", max_len={arg.token_len}, '
        f'vocab_path="{vocab_path}", add_bos=True, add_eos=True)'
    )

  # Zero-shot ImageNet classification
  config.evals.disclf = {}
  config.evals.disclf.dataset_names = ['imagenet2012']
  config.evals.disclf.split = f'validation{sub}'
  config.evals.disclf.data_dir = 'gs://your-tfds-bucket/imagenet2012'  # placeholder
  config.evals.disclf.pp_img = (
      f'|resize_small({arg.res}, method="bilinear", antialias=True)'
      f'|central_crop({arg.res})|vgg_value_range'
  )
  config.evals.disclf.pp_txt = tokenizer_eval('texts')
  config.evals.disclf.canonicalize = True
  config.evals.disclf.first_class_name_only = False
  config.evals.disclf.type = 'proj.image_text.discriminative_classifier'
  config.evals.disclf.prefix = 'z/0shot/'
  config.evals.disclf.log_steps = eval_common['log_steps']

  # COCO retrieval
  config.evals.retrieval = dict(
      log_steps=eval_common['log_steps'],
      type='proj.image_text.retrieval',
      dataset='coco_captions',
      split='val',
      data_dir='gs://your-bucket/coco',  # placeholder
      txt_name=('captions', 'text'),
      pp_img=(
          f'|resize_small({arg.res}, method="bilinear", antialias=True)'
          f'|central_crop({arg.res})|vgg_value_range'
      ),
      pp_txt=tokenizer_eval('texts'),
  )

  # Flickr30k retrieval
  config.evals.retrieval_flikr = dict(
      log_steps=eval_common['log_steps'],
      type='proj.image_text.retrieval',
      dataset='flickr30k',
      split='test',
      data_dir='gs://your-bucket/flickr30k',  # placeholder
      txt_name='captions',
      pp_img=(
          f'|resize_small({arg.res}, method="bilinear", antialias=True)'
          f'|central_crop({arg.res})|vgg_value_range'
      ),
      pp_txt=tokenizer_eval('texts'),
  )

  #####################################
  #         misc and HF upload        #
  #####################################
  config.seed = 0
  config.l = 0
  config.m = 0

  # Optional: publish to Hugging Face. Keep placeholders only.
  config.hf_upload = dict(
      repo_name='your-hf-repo-name',
      save_directory='/path/on/vm/to/save',
      token='your-hf-token',
      commit_message='initial commit',
      model_id='your-namespace/ov2-vitL14-224',  # template model id
      cache_dir='/path/to/hf_cache',
  )

  return config
