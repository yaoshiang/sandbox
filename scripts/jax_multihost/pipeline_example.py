"""Minimal 2-stage pipeline-parallel example across hosts.

Pipeline stage 0 runs on the first half of the slice (hosts 0 & 1, 8 TPUs).
Pipeline stage 1 runs on the second half of the slice (hosts 2 & 3, 8 TPUs).

Every process runs this EXACT same script, making it SPMD-ish.

But notice the crucial MPMD behavior: 
- Pipeline stage 1 hosts run the same program as pipeline stage 0 hosts...
- but stage 1 hosts/TPUs do not run ops mapped to pipeline stage 0's mesh.
- Device transfer is through a jax.device_put. This is an incredibly powerful, 
  single-controller-ish paradigm. While it is not true single-controller, 
  it does allow each host to address non-local TPUs directly. 
"""
import contextlib
import os
import sys
# 1. Suppress libtpu / XLA / TensorFlow C++ logs (0=INFO, 1=WARNING, 2=ERROR, 3=FATAL)
# Must be set BEFORE importing jax.
os.environ["TPU_STDERR_LOG_LEVEL"] = "3"
os.environ["TPU_MIN_LOG_LEVEL"] = "3"
os.environ["TPU_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import jax
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

jax.distributed.initialize()

me = jax.process_index()
num_devices = jax.device_count() // 2


def log(msg):
    print(f"[host {me}] {msg}", flush=True)


log(f"Total slice devices: {jax.device_count()}, devices per stage: {num_devices}")

# Mesh for Stage 0 (first half: hosts 0 & 1, 8 chips forming a valid physical sub-slice)
mesh_stage0 = jax.make_mesh(
    (num_devices,),
    ('a',),
    devices=jax.devices()[:num_devices],
    axis_types=(jax.sharding.AxisType.Explicit,)
)
sharding_stage0 = NamedSharding(mesh_stage0, P('a'))

# Mesh for Stage 1 (second half: hosts 2 & 3, 8 chips forming a valid physical sub-slice)
mesh_stage1 = jax.make_mesh(
    (num_devices,),
    ('a',),
    devices=jax.devices()[num_devices:],
    axis_types=(jax.sharding.AxisType.Explicit,)
)
sharding_stage1 = NamedSharding(mesh_stage1, P('a'))

# Initial data: simple 1D array [0, 1, 2, 3, 4, 5, 6, 7]
data = np.arange(8, dtype=np.float32)

# Step 1a: input
x = jax.device_put(data, sharding_stage0)
log(f"Step 1a (input): addressable_shards={len(x.addressable_shards)} data={x.addressable_shards[0].data if len(x.addressable_shards) > 0 else 'none'}")

# Step 1b: math op
@jax.jit
def f(x):
    return x + 1.0


y = f(x)
log(f"Step 1b (math op): addressable_shards={len(y.addressable_shards)} data={y.addressable_shards[0].data if len(y.addressable_shards) > 0 else 'none'}")

# Step 2a: transfer
z = jax.device_put(y, sharding_stage1)
z.block_until_ready()
log(f"Step 2a (transfer): addressable_shards={len(z.addressable_shards)} data={z.addressable_shards[0].data if len(z.addressable_shards) > 0 else 'none'}")

# Step 2b: math op
@jax.jit
def g(z):
    return z * 2.0


result = g(z)
result.block_until_ready()
log(f"Step 2b (math op): addressable_shards={len(result.addressable_shards)} data={result.addressable_shards[0].data if len(result.addressable_shards) > 0 else 'none'}")

# Explicit clean shutdown barrier
jax.distributed.shutdown()
