"""This script demonstrates propagation and partitioning in JAX.

To run it, you'll need to pip install JAX cpu. It's probably safe to do so
over the torch_titan based requirements.txt which set up a decent PyTorch CUDA environment.

```
pip install --upgrade "jax[cpu]"
```

This script helps delineate multiple interrelated concepts.

Propagation: Given two sharded input tensors, what's the right sharding for the output tensor?
This is obvious for simple ops, but for example at the end of the Megatron style TP sharding
of a 2 layer FF network, should the output be sharded along the TP axis or replicated? The
original paper says replicated, but the JAX Scaling playbook recommends sharding along
the TP axis. *Someone* has to decide this.

In [the JAX docs](https://docs.jax.dev/en/latest/notebooks/explicit-sharding.html#using-a-mixture-of-sharding-modes),
The 1st form of propagation is Automatic. In Automatic, the compiler
handles output shardings... this is referred to as compiler tickling and is generally not fun.

The 2nd form of propagation mentioned is Explicit. That means, either JAX decides
on the output sharding based on simple, obvious rules. When such a rule
does not exist, JAX requires that the programmer specify an out_sharding.

The 3rd and final form is basically manual collectives - the programmer figures it out.

Partitioning: Given a sharding for the inputs and outputs, how should the computation be arranged?
For example, when adding two sharded tensors, we can
all gather first, then sum, or, get a partial sum, then all-reduce. That decision is
called "partitioning" in the language of GSPMD and PartIR and Shardy.

[GSPMD](https://arxiv.org/abs/2105.04663): An older platform to handle propagation and partitioning.

[PartIR](https://arxiv.org/abs/2401.11202): Yet another approach, out of GDM.

[Shardy](https://openxla.org/shardy): A newer one built by the teams behind both GSPMD and PartIR.

The following script demonstrates Explicit propagation on Megatron sharding. We specify
an out_sharding, forcing this to be explicit. It's still Shardy/GSPMD that
decides "partitioning", or the order of math ops and collectives to get to the right answer.
Here, it does NOT use reduce-scatter, but instead uses all-reduce and slice, likely because
this is run on a mocked CPU mesh.

Output below. Note that the compiled SHLO function takes smaller tensors than x, ff1, or ff2.
This is because GSPMD/Shardy has figured out how to shard the jax.Arrays into
per-device shards.


```txt
WARNING:2026-01-09 00:24:02,791:jax._src.xla_bridge:852: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
jax.devices()=[CpuDevice(id=0), CpuDevice(id=1), CpuDevice(id=2), CpuDevice(id=3), CpuDevice(id=4), CpuDevice(id=5), CpuDevice(id=6), CpuDevice(id=7)]
mesh=Mesh(axis_sizes=(2, 4), axis_names=('dp', 'tp'), axis_types=(Auto, Auto))

x=Array([[   0,    1,    2, ..., 1021, 1022, 1023],
       [1024, 1025, 1026, ..., 2045, 2046, 2047],
       [2048, 2049, 2050, ..., 3069, 3070, 3071],
       [3072, 3073, 3074, ..., 4093, 4094, 4095]], dtype=int32)

ff1=Array([[      0,       1,       2, ...,    4093,    4094,    4095],
       [   4096,    4097,    4098, ...,    8189,    8190,    8191],
       [   8192,    8193,    8194, ...,   12285,   12286,   12287],
       ...,
       [4182016, 4182017, 4182018, ..., 4186109, 4186110, 4186111],
       [4186112, 4186113, 4186114, ..., 4190205, 4190206, 4190207],
       [4190208, 4190209, 4190210, ..., 4194301, 4194302, 4194303]],      dtype=int32)

ff2=Array([[      0,       1,       2, ...,    1021,    1022,    1023],
       [   1024,    1025,    1026, ...,    2045,    2046,    2047],
       [   2048,    2049,    2050, ...,    3069,    3070,    3071],
       ...,
       [4191232, 4191233, 4191234, ..., 4192253, 4192254, 4192255],
       [4192256, 4192257, 4192258, ..., 4193277, 4193278, 4193279],
       [4193280, 4193281, 4193282, ..., 4194301, 4194302, 4194303]],      dtype=int32)

x.sharding=NamedSharding(mesh=Mesh('dp': 2, 'tp': 4, axis_types=(Auto, Auto)), spec=PartitionSpec('dp', 'tp'), memory_kind=device)

ff1.sharding=NamedSharding(mesh=Mesh('dp': 2, 'tp': 4, axis_types=(Auto, Auto)), spec=PartitionSpec(None, 'tp'), memory_kind=device)

ff2.sharding=NamedSharding(mesh=Mesh('dp': 2, 'tp': 4, axis_types=(Auto, Auto)), spec=PartitionSpec('tp', None), memory_kind=device)

jax.typeof(x)=ShapedArray(int32[4,1024])

jax.typeof(ff1)=ShapedArray(int32[1024,4096])

jax.typeof(ff2)=ShapedArray(int32[4096,1024])

jax.eval_shape(f, x, ff1, ff2)=ShapeDtypeStruct(shape=(4, 1024), dtype=int32)

lowered.as_text()= module @jit_f attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["dp"=2, "tp"=4]>
  func.func public @main(%arg0: tensor<4x1024xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"dp"}, {"tp"}]>}, %arg1: tensor<1024x4096xi32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"tp"}]>}, %arg2: tensor<4096x1024xi32> {sdy.sharding = #sdy.sharding<@mesh, [{"tp"}, {}]>}) -> (tensor<4x1024xi32> {jax.result_info = "result", sdy.sharding = #sdy.sharding<@mesh, [{"dp"}, {"tp"}]>}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x1024xi32>, tensor<1024x4096xi32>) -> tensor<4x4096xi32>
    %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x4096xi32>, tensor<4096x1024xi32>) -> tensor<4x1024xi32>
    return %1 : tensor<4x1024xi32>
  }
}


computation.as_text()= HloModule jit_f, is_scheduled=true, entry_computation_layout={(s32[2,256]{1,0}, s32[1024,1024]{1,0}, s32[1024,1024]{1,0})->s32[2,256]{1,0}}, allow_spmd_sharding_propagation_to_parameters={false,false,false}, num_partitions=8

FileNames
1 "/home/yho_google_com/Documents/GitHub/sandbox/scripts/jax_shardy_propagation_partitioning.py"

FunctionNames
1 "<module>"
2 "main"
3 "main.<locals>.f"

FileLocations
1 {file_name_id=1 function_name_id=1 line=85 end_line=85 column=13 end_column=19}
2 {file_name_id=1 function_name_id=2 line=65 end_line=65 column=19 end_column=49}
3 {file_name_id=1 function_name_id=3 line=63 end_line=63 column=20 end_column=29}
4 {file_name_id=1 function_name_id=3 line=63 end_line=63 column=19 end_column=37}

StackFrames
1 {file_location_id=1 parent_frame_id=1}
2 {file_location_id=2 parent_frame_id=2}
3 {file_location_id=3 parent_frame_id=3}
4 {file_location_id=4 parent_frame_id=3}


%add.clone (x.1: s32[], y.1: s32[]) -> s32[] {
  %x.1 = s32[] parameter(0)
  %y.1 = s32[] parameter(1)
  ROOT %add.1 = s32[] add(%x.1, %y.1)
}

%fused_computation (param_0: s32[2,1024], param_1.1: u32[8], param_2.6: u32[]) -> s32[2,256] {
  %param_0 = s32[2,1024]{1,0} parameter(0)
  %constant.18 = s32[] constant(0), metadata={op_name="jit(f)/dot_general" stack_frame_id=4}
  %constant.17 = u32[1]{0} constant({0}), metadata={op_name="jit(f)/dot_general" stack_frame_id=4}
  %param_1.1 = u32[8]{0} parameter(1)
  %param_2.6 = u32[] parameter(2)
  %dynamic-slice.8 = u32[1]{0} dynamic-slice(%param_1.1, %param_2.6), dynamic_slice_sizes={1}, metadata={op_name="jit(f)/dot_general" stack_frame_id=4}
  %constant.16 = u32[1]{0} constant({3}), metadata={op_name="jit(f)/dot_general" stack_frame_id=4}
  %clamp.2 = u32[1]{0} clamp(%constant.17, %dynamic-slice.8, %constant.16), metadata={op_name="jit(f)/dot_general" stack_frame_id=4}
  %convert.2 = s32[1]{0} convert(%clamp.2), metadata={op_name="jit(f)/dot_general" stack_frame_id=4}
  %constant.15 = s32[1]{0} constant({256}), metadata={op_name="jit(f)/dot_general" stack_frame_id=4}
  %multiply.3 = s32[1]{0} multiply(%convert.2, %constant.15), metadata={op_name="jit(f)/dot_general" stack_frame_id=4}
  %bitcast.1 = s32[] bitcast(%multiply.3), metadata={op_name="jit(f)/dot_general" stack_frame_id=4}
  ROOT %dynamic-slice.7 = s32[2,256]{1,0} dynamic-slice(%param_0, %constant.18, %bitcast.1), dynamic_slice_sizes={2,256}, metadata={op_name="jit(f)/dot_general" stack_frame_id=4}
}

ENTRY %main.0_spmd (param: s32[2,256], param.1: s32[1024,1024], param.2: s32[1024,1024]) -> s32[2,256] {
  %partition-id = u32[] partition-id()
  %param = s32[2,256]{1,0} parameter(0), sharding={devices=[2,4]<=[8]}, metadata={op_name="x_"}
  %param.1 = s32[1024,1024]{1,0} parameter(1), sharding={devices=[1,4,2]<=[2,4]T(1,0) last_tile_dim_replicate}, metadata={op_name="ff1_"}
  %param.2 = s32[1024,1024]{1,0} parameter(2), sharding={devices=[4,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}, metadata={op_name="ff2_"}
  %constant.4 = u32[8]{0} constant({0, 1, 2, 3, 0, 1, 2, 3}), metadata={op_name="jit(f)/dot_general" stack_frame_id=4}
  %copy = s32[2,256]{0,1} copy(%param), sharding={devices=[2,4]<=[8]}, metadata={op_name="x_"}
  %all-gather = s32[2,1024]{0,1} all-gather(%copy), channel_id=1, replica_groups=[2,4]<=[8], dimensions={1}, use_global_device_ids=true, metadata={op_name="jit(f)/dot_general" stack_frame_id=3}
  %copy.1 = s32[2,1024]{1,0} copy(%all-gather), metadata={op_name="jit(f)/dot_general" stack_frame_id=3}
  %dot = s32[2,1024]{1,0} dot(%copy.1, %param.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="jit(f)/dot_general" stack_frame_id=3}
  %dot.1 = s32[2,1024]{1,0} dot(%dot, %param.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_name="jit(f)/dot_general" stack_frame_id=4}
  %all-reduce = s32[2,1024]{1,0} all-reduce(%dot.1), channel_id=2, replica_groups=[2,4]<=[8], use_global_device_ids=true, to_apply=%add.clone, metadata={op_name="jit(f)/dot_general" stack_frame_id=4}
  ROOT %bitcast_dynamic-slice_fusion = s32[2,256]{1,0} fusion(%all-reduce, %constant.4, %partition-id), kind=kLoop, calls=%fused_computation, metadata={op_name="jit(f)/dot_general" stack_frame_id=4}
}


NamedSharding(mesh=Mesh('dp': 2, 'tp': 4, axis_types=(Auto, Auto)), spec=PartitionSpec('dp', 'tp'), memory_kind=device)
```
"""

import os
import sys

os.environ["XLA_TPU_ENABLE_REDUCE_SCATTER"] = "true"

import jax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import jax.numpy as jnp
import numpy as np


def main() -> int:
    jax.config.update("jax_num_cpu_devices", 8)

    print(f"{jax.devices()=}")
    assert (
        str(jax.devices()) == "[CpuDevice(id=0), CpuDevice(id=1), "
        "CpuDevice(id=2), CpuDevice(id=3), CpuDevice(id=4), CpuDevice(id=5), CpuDevice(id=6), CpuDevice(id=7)]"
    )

    # Create a mesh of shape 2,4.
    mesh = Mesh(np.array(jax.devices()).reshape(2, 4), ("dp", "tp"))
    print(f"{mesh=}")

    with jax.set_mesh(mesh):
        bsz = 4
        hidden = 1024
        up_dim = hidden * 4

        x = jnp.arange(bsz * hidden).reshape(bsz, hidden)
        ff1 = jnp.arange(hidden * up_dim).reshape(hidden, up_dim)
        ff2 = jnp.arange(up_dim * hidden).reshape(up_dim, hidden)

        x = jax.device_put(x, NamedSharding(mesh, P("dp", "tp")))
        ff1 = jax.device_put(ff1, NamedSharding(mesh, P(None, "tp")))
        ff2 = jax.device_put(ff2, NamedSharding(mesh, P("tp", None)))

        print(f"\n{x=}")
        print(f"\n{ff1=}")
        print(f"\n{ff2=}")

        print(f"\n{x.sharding=}")
        print(f"\n{ff1.sharding=}")
        print(f"\n{ff2.sharding=}")

        print(f"\n{jax.typeof(x)=}")
        print(f"\n{jax.typeof(ff1)=}")
        print(f"\n{jax.typeof(ff2)=}")

        def f(x_, ff1_, ff2_):
            return (x_ @ ff1_) @ ff2_

        print(f"\n{jax.eval_shape(f, x, ff1, ff2)=}")

        jitted_f = jax.jit(
            f,
            in_shardings=(x.sharding, ff1.sharding, ff2.sharding),
            out_shardings=x.sharding,
        )
        lowered = jitted_f.lower(x, ff1, ff2)
        print("\nlowered.as_text()=", lowered.as_text())

        computation = lowered.compile()
        print("\ncomputation.as_text()=", computation.as_text())

        c = jitted_f(x, ff1, ff2)
        print(c.sharding)

    return 0


if __name__ == "__main__":
    sys.exit(main())
