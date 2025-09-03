# Copyright 2023 Yanqing Liu
#
# This code is based on materials from Big Vision (https://github.com/google-research/big_vision).
# Licensed under the Apache License, Version 2.0.

"""OpenVision 2 training entry (sanitized for open-source).

Supports large-scale JAX/TPU training with sharded params/opt state,
captioning loss (CoCa-style), and retrieval/classification evals.
"""

from absl import app, flags, logging
from ml_collections import config_flags
import functools
import importlib
import multiprocessing.pool
import os
import sys

# JAX/Flax/Optax
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from flax.linen import partitioning as nn_partitioning
from flax import linen as nn
import optax

# TF input pipeline backend
import tensorflow as tf

# Project imports (keep interfaces unchanged)
from transforms.mixup import mixup, cutmix  # noqa: F401 (kept for future use)
from helpers.utils import *  # noqa: F401,F403 (chrono, metrics, ckpt utils, etc.)
from helpers.sharding import *  # noqa: F401,F403 (create_mesh, reshard helpers)
from datasets import input_pipeline
from datasets.input_pipeline import shard_and_put  # noqa: F401
from clu import parameter_overview
import optim
import losses
import evaluators.common as eval_common

# Optional W&B
try:
    import wandb
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False
    print("wandb not installed. To enable logging: pip install wandb")

# -----------------------------
# Flags
# -----------------------------
config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", default=None, help="Work unit directory for checkpoints and logs.")
flags.DEFINE_boolean("cleanup", default=False, help="Delete workdir after successful completion.")
flags.DEFINE_bool("disable_gpu", default=True, help="If True, hide local GPUs for TF input pipeline.")
FLAGS = flags.FLAGS

# Parse JAX flags early
jax.config.parse_flags_with_absl()
jax.config.update("jax_threefry_partitionable", True)  # reduce d2d comms in PRNG


def main(argv):
    del argv

    # -----------------------------
    # Multi-host init and env setup
    # -----------------------------
    jax.distributed.initialize()
    jax.config.update("jax_default_prng_impl", "unsafe_rbg")  # match original behavior

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "0")
    if FLAGS.disable_gpu:
        # Avoid TF grabbing GPUs on TPU machines
        try:
            tf.config.experimental.set_visible_devices([], "GPU")
        except Exception:
            pass

    config = FLAGS.config
    workdir = FLAGS.workdir
    logging.info(
        f"\033[33mHello from process {jax.process_index()} holding "
        f"{jax.local_device_count()}/{jax.device_count()} devices and "
        f"writing to workdir {workdir}.\033[0m"
    )

    save_ckpt_path = None
    if workdir:
        tf.io.gfile.makedirs(workdir)
        save_ckpt_path = os.path.join(workdir, "checkpoint.npz")

    # ThreadPool for async logging and small IO
    pool = multiprocessing.pool.ThreadPool()

    # Register preprocessing ops
    for m in config.get("pp_modules", ["ops_general", "ops_image", "ops_text"]):
        importlib.import_module(f"transforms.{m}")

    # RNG
    rng = jax.random.PRNGKey(
        jax.device_put(config.get("seed", 0), jax.local_devices(backend="cpu")[0])
    )

    # -----------------------------
    # Logging setup (W&B optional)
    # -----------------------------
    xid, wid = -1, -1

    def info(s, *a):
        logging.info("\033[33mNOTE\033[0m: " + s, *a)

    def write_note(note):
        if jax.process_index() == 0:
            info("%s", note)

    write_note("Initializing logging...")
    if config.wandb.get("log_wandb", False):
        if HAS_WANDB and jax.process_index() == 0:
            if config.wandb.get("wandb_offline", False):
                os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project=str(config.wandb.get("project", "openvision2")),
                name=str(config.wandb.get("experiment", "exp")),
                entity=str(config.wandb.get("entity", "")) or None,
                resume=bool(config.wandb.get("resume", False)),
            )
            wandb.config.update(dict(config))
        else:
            logging.warning("wandb requested but not installed or not on host 0.")

    metric = BigVisionMetricWriter(xid, wid, workdir, config)

    # -----------------------------
    # Device mesh and sharding
    # -----------------------------
    write_note("Creating mesh...")
    device_arrays = create_mesh(config)
    mesh = Mesh(device_arrays, config.sharding.mesh_axes)
    data_sharding = jax.tree.map(lambda p: jax.sharding.NamedSharding(mesh, p), P(*config.sharding.data_sharding))
    repl_sharding = jax.sharding.NamedSharding(mesh, P())

    # A 1D replicated mesh for certain utilities
    repl_mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), axis_names=("repl",))

    # -----------------------------
    # Dataset
    # -----------------------------
    write_note("Initializing train dataset...")
    batch_size = config.input.batch_size
    if batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size ({batch_size}) must be divisible by device number ({jax.device_count()})")
    info(
        "Global batch size %d on %d hosts => local batch %d; %d dev/host (%d total) => per-device %d",
        batch_size,
        jax.process_count(),
        batch_size // jax.process_count(),
        jax.local_device_count(),
        jax.device_count(),
        batch_size // jax.device_count(),
    )

    tokenizer = None
    if config.get("openclip_tokenizer.enable", False):
        try:
            import open_clip  # lazy import
        except Exception:
            raise ImportError("Please `pip install open_clip_torch` to use OpenCLIP tokenizer.")
        tokenizer = open_clip.get_tokenizer(f'hf-hub:{config.openclip_tokenizer.repo_name}')

    train_ds, ntrain_img = input_pipeline.training(config.input)
    train_iter = input_pipeline.start_input_pipeline(
        train_ds, config=config, mesh=mesh, data_sharding=data_sharding, tokenizer=tokenizer
    )

    # Steps and chrono
    total_steps = steps("total", config, ntrain_img, batch_size)
    get_steps = lambda name, default=ValueError, cfg=config: steps(name, cfg, ntrain_img, batch_size, total_steps, default)
    chrono.inform(
        total_steps=total_steps,
        global_bs=batch_size,
        steps_per_epoch=ntrain_img / batch_size,
        measure=metric.measure,
        write_note=write_note,
    )
    info("Running for %d steps, equals %f epochs", total_steps, total_steps * batch_size / ntrain_img)

    # -----------------------------
    # Model, optimizer, scheduler
    # -----------------------------
    write_note(f"Initializing {config.model_name}...")
    model_mod = importlib.import_module(f"models.{config.model_name}")
    model = model_mod.Model(**config.get("model", {}), mesh=mesh)

    def init(rng_key):
        image_shape = config.init_shapes[0]
        text_shape = config.init_shapes[1]
        no_image = jnp.zeros(image_shape, jnp.float32)
        no_text = jnp.zeros(text_shape, jnp.int32)
        params = model.init(rng_key, no_image, no_text)["params"]
        return params

    write_note("Inferring parameter shapes...")
    rng, rng_init = jax.random.split(rng)

    # logical -> mesh annotations for params and optimizer state
    with nn_partitioning.axis_rules(config.sharding.logical_axis_rules):
        params_shape = jax.eval_shape(init, rng_init)
    params_logical = nn.get_partition_spec(params_shape)
    with mesh, nn_partitioning.axis_rules(config.sharding.logical_axis_rules):
        params_mesh = nn.logical_to_mesh(params_logical)
    params_mesh_shardings = nn.logical_to_mesh_sharding(params_logical, mesh, config.sharding.logical_axis_rules)
    params_unboxed_shape = unbox_logicallypartioned(params_shape)

    if jax.process_index() == 0:
        num_params = sum(p.size for p in jax.tree.leaves(params_shape))
        metric.measure("num_params", num_params)

    write_note(f"Initializing optimizer {config.optax_name}...")
    tx, sched_fns = optim.make(
        config, params_unboxed_shape, sched_kw=dict(total_steps=total_steps, batch_size=batch_size, data_size=ntrain_img)
    )
    with nn_partitioning.axis_rules(config.sharding.logical_axis_rules):
        opt_shape = jax.eval_shape(tx.init, params_shape)
    opt_logical = nn.get_partition_spec(opt_shape)
    with mesh, nn_partitioning.axis_rules(config.sharding.logical_axis_rules):
        opt_mesh = nn.logical_to_mesh(opt_logical)
    opt_mesh_shardings = nn.logical_to_mesh_sharding(opt_logical, mesh, config.sharding.logical_axis_rules)

    # Compile schedulers for CPU
    sched_fns_cpu = [jax.jit(s, backend="cpu") for s in sched_fns]

    train_state_sharding = {
        "params": jax.tree_util.tree_map(lambda p: jax.sharding.NamedSharding(mesh, p), params_mesh),
        "opt": jax.tree_util.tree_map(lambda p: jax.sharding.NamedSharding(mesh, p), opt_mesh),
    }

    write_note("Transferring train_state to devices...")
    rng_init = reshard(rng_init, repl_sharding)
    params = jax.jit(init, in_shardings=None, out_shardings=params_mesh_shardings)(rng_init)
    opt = jax.jit(tx.init, out_shardings=opt_mesh_shardings)(params)

    # Unbox partitioned structures for custom WD etc.
    params = unbox_logicallypartioned(params)
    opt = unbox_logicallypartioned(opt)

    rng, rng_loop = jax.random.split(rng, 2)
    rng_loop = reshard(rng_loop, repl_sharding)
    train_state = {"params": params, "opt": opt}
    del params, opt

    parameter_overview.log_parameter_overview(train_state["params"], msg="Init params", include_stats="global", jax_logging_process=0)

    # -----------------------------
    # Update step (jit)
    # -----------------------------
    @functools.partial(
        jax.jit,
        donate_argnums=(0,),
        in_shardings=(train_state_sharding, data_sharding, repl_sharding),
        out_shardings=(train_state_sharding, repl_sharding),
    )
    def update_fn(state, batch, rng_key):
        images = batch["image"]
        labels = batch["labels"]

        if config.get("cpu_unit8", False):
            mean = jnp.asarray([0.485 * 255, 0.456 * 255, 0.406 * 255])[None, None, None, :]
            std = jnp.asarray([0.229 * 255, 0.224 * 255, 0.225 * 255])[None, None, None, :]
            images = (jnp.asarray(images, dtype=jnp.float32) - mean) / std

        step_count = optim.get_count(state["opt"], jittable=True)
        rng_key = jax.random.fold_in(rng_key, step_count)

        def loss_fn(params_, imgs, lbls):
            extras = model.apply({"params": params_}, imgs, lbls, train=True, rngs={"dropout": rng_key, "drop_path": rng_key, "random_mask": rng_key})

            # autoregressive captioning loss
            autoreg_labels = batch["autoreg_labels"]
            logits_txt = extras["logits"]
            cap_loss_mask = batch["cap_loss_mask"]
            caption_l = losses.softmax_xent(logits=logits_txt, labels=autoreg_labels, mask=cap_loss_mask, reduction=True, axis=-1)

            cap_w = config.get("coca_caption_loss_weight", 2.0)
            l = cap_w * caption_l
            return l, {"caption_loss": caption_l}

        (loss_val, measurements), grads = jax.value_and_grad(loss_fn, has_aux=True)(state["params"], images, labels)
        updates, opt_ = tx.update(grads, state["opt"], state["params"])
        params_ = optax.apply_updates(state["params"], updates)

        measurements["training_loss"] = loss_val
        gs = jax.tree.leaves(optim.replace_frozen(config.schedule, grads, 0.0))
        measurements["l2_grads"] = jnp.sqrt(sum([jnp.vdot(g, g) for g in gs]))
        ps = jax.tree.leaves(params_)
        measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in ps]))
        us = jax.tree.leaves(updates)
        measurements["l2_updates"] = jnp.sqrt(sum([jnp.vdot(u, u) for u in us]))

        return {"params": params_, "opt": opt_}, measurements

    # -----------------------------
    # Checkpoint loading / resume
    # -----------------------------
    resume_ckpt_path = None
    if save_ckpt_path and tf.io.gfile.exists(f"{save_ckpt_path}"):
        resume_ckpt_path = save_ckpt_path
    elif config.get("resume"):
        resume_ckpt_path = fillin(config.resume)

    ckpt_mngr = None
    if save_ckpt_path or resume_ckpt_path:
        ckpt_mngr = create_orbax_checkpoint_manager(
            save_ckpt_path, create=True, async_checkpoints=True, save_interval_steps=1, max_to_keep=1
        )

    if ckpt_mngr:
        latest_step = ckpt_mngr.latest_step()
        if latest_step:
            write_note(f"Resuming from checkpoint step {latest_step}...")
            abstract_state = jax.tree_util.tree_map(orbax.checkpoint.utils.to_shape_dtype_struct, train_state)
            train_state = ckpt_mngr.restore(latest_step, args=orbax.checkpoint.args.StandardRestore(abstract_state))

            chrono_ckpt_path = save_ckpt_path.replace("checkpoint.npz", "chrono.npz")
            chrono_loaded = load_checkpoint({"chrono": chrono.save()}, chrono_ckpt_path)
            chrono.load(chrono_loaded["chrono"])
        elif config.get("ft_from", None):
            ckpt_path = config.get("ft_from")
            write_note(f"Initialize params from {ckpt_path}...")
            abstract_state = jax.tree_util.tree_map(orbax.checkpoint.utils.to_shape_dtype_struct, train_state)
            ft_mngr = create_orbax_checkpoint_manager(ckpt_path, create=True, async_checkpoints=True, save_interval_steps=1, max_to_keep=1)
            latest_step = ft_mngr.latest_step()
            loaded = ft_mngr.restore(latest_step, args=orbax.checkpoint.args.StandardRestore(abstract_state))
            train_state["params"] = loaded["params"]
            parameter_overview.log_parameter_overview(train_state["params"], msg="Restored params", include_stats="global", jax_logging_process=0)
        elif config.get("masked_init"):
            write_note(f"Masked init from {config.masked_init}...")
            pretrained_params_cpu = load_params(None, config.masked_init)
            params_cpu = jax.tree.map(recover_dtype, pretrained_params_cpu)
            train_state["params"] = reshard(params_cpu, params_mesh_shardings)

    # -----------------------------
    # Evaluators
    # -----------------------------
    def eval_logits_fn(state, batch):
        zimg, ztxt, out = model.apply({"params": state["params"]}, batch.get("image", None), batch.get("labels", None))
        return zimg, ztxt, out

    eval_fns = {"predict": eval_logits_fn}

    @functools.lru_cache(maxsize=None)
    def evaluators():
        return eval_common.from_config(
            config,
            eval_fns,
            lambda s: write_note(f"Init evaluator: {s}…\n{chrono.note}"),
            lambda key, cfg: get_steps(key, default=None, cfg=cfg),
            mesh,
            data_sharding,
            train_state_sharding["params"],
            tokenizer=tokenizer,
        )

    # -----------------------------
    # Training loop
    # -----------------------------
    write_note("Kicking off training...")
    first_step_device = optim.get_count(train_state["opt"], jittable=True)
    first_step = int(jax.device_get(first_step_device))
    chrono.inform(first_step=first_step)
    prof = None

    if config.get("eval_only", False):
        step = 0
        for (name, evaluator, _, prefix) in evaluators():
            chrono.pause(wait_for=train_state)
            chrono.tick(step)
            write_note(f"{name} evaluation...\n{chrono.note}")
            with chrono.log_timing(f"z/secs/eval/{name}"):
                with mesh, nn.logical_axis_rules(config.sharding.logical_axis_rules):
                    for key, value in evaluator.run(train_state):
                        metric.measure(f"{prefix}{key}", value)
            chrono.resume()
        metric.step_end()
        return

    write_note(f"Compiling first steps...\n{chrono.note}")
    for step, batch in zip(range(first_step + 1, total_steps + 1), train_iter):
        metric.step_start(step)
        jax.experimental.multihost_utils.sync_global_devices("data_loading")

        with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
            with chrono.log_timing("z/secs/update0", noop=step > first_step + 1):
                with mesh, nn.logical_axis_rules(config.sharding.logical_axis_rules):
                    train_state, measurements = update_fn(train_state, batch, rng_loop)

        if jax.process_index() == 0:
            prof = startstop_prof(prof, step, first_step, get_steps("log_training"))

        if (itstime(step, get_steps("log_training"), total_steps, host=0) or chrono.warmup and jax.process_index() == 0):
            for i, sched_fn_cpu in enumerate(sched_fns_cpu):
                metric.measure(
                    f"global_schedule{i or ''}",
                    sched_fn_cpu(jax.device_put(step - 1, jax.local_devices(backend="cpu")[0])),
                )
            measurements = jax.device_get(measurements)
            for name, value in measurements.items():
                metric.measure(name, value)
            chrono.tick(step)

        jax.experimental.multihost_utils.sync_global_devices("reporting")

        # Save checkpoints
        keep_ckpt_steps = get_steps("keep_ckpt", None) or total_steps
        if save_ckpt_path and (
            (keep := itstime(step, keep_ckpt_steps, total_steps, first=False))
            or itstime(step, get_steps("ckpt", None), total_steps, first=True)
        ):
            chrono.pause(wait_for=(train_state))
            ckpt = {**train_state}
            ckpt_mngr.save(step, args=orbax.checkpoint.args.StandardSave(ckpt))
            jax.experimental.multihost_utils.sync_global_devices("final_eval")

            chrono_ckpt_path = save_ckpt_path.replace("checkpoint.npz", "chrono.npz")
            chronockpt = {"chrono": chrono.save()}
            pool.apply_async(save_checkpoint, (chronockpt, chrono_ckpt_path, None))
            chrono.resume()

        metric.step_end()
        jax.experimental.multihost_utils.sync_global_devices("eval")

        if HAS_WANDB and jax.process_index() == 0 and config.wandb.get("log_wandb", False):
            wandb.log(metric.step_metrics, step=step)
        jax.experimental.multihost_utils.sync_global_devices("wandb_log")

    # Final eval/log
    metric.step_start(total_steps)
    if HAS_WANDB and jax.process_index() == 0 and config.wandb.get("log_wandb", False):
        wandb.log(metric.step_metrics, step=total_steps)
    if jax.process_index() == 0 and prof is not None:
        startstop_prof(prof)
    jax.experimental.multihost_utils.sync_global_devices("final_eval")

    write_note(f"Done!\n{chrono.note}")
    pool.close()
    pool.join()
    metric.close()
    maybe_cleanup_workdir(workdir, FLAGS.cleanup, info)
    jax.experimental.multihost_utils.sync_global_devices("done")


if __name__ == "__main__":
    app.run(main)
