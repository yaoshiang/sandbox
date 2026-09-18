# Multihost, single-controller-ish, MPMD-ish behavior in JAX

JAX has a fascinating approach to performing what PyTorch / MPI would call a send/recv, but in a way that still feels like SPMD. The key APIs that enable this:
 
1. The creation of multiple meshes, each of which span less than the full set of available devices
1. All ranks see all code that runs on all ranks, even if that code does not actually run on the local rank
1. jax.device_put on a jax.Array living on one mesh and assigned to a second mesh will be treated as if it were a send/recv (or even more powerfully, equivalent to a pathways.utils.reshard). 

## Problem and Solution

Directionally, JAX focuses users towards sharded arrays which enforce SPMD behavior. Even “raw” collectives inside a shard_map have input and output spmd typing. JAX doesn’t have send and recv. How might an MPMD pipeline parallel solution be implemented?

The solution is cross-host `jax.device_put`. The secret to making crosshost `jax.device_put` look SPMD-ish is to have all ranks run all the code, but the framework to internally create no-ops when the local rank is not participating. 

Conceptually, imagine two hosts, each with one TPU. Each runs this code. 

```python
# jax.devices() returns a list of global devices across all hosts
[device_0, device_1] = jax.devices()

data = jnp.ones((1,), device=device_0)
data = data + 1.0
data = jax.device_put(data, device=device_1)
data = data * 2.0
```

Both ranks participate in the `jax.device_put` call. For device_0 it is basically a send collective. For device_1, it is basically a recv collective. 

Notice that device_1 is aware of device_0’s buffer allocation (the first jnp.ones). But device_1 will not allocate any real data - it silently observes and is aware of device_0’s work. 

Both ranks participate in the jax.device_put call. For device_0 it is basically a send collective. For device_1, it is basically a recv collective. 

Similarly, after the jax.device_put call, device_0 will be aware that device_1 owns and accesses the data, but it does not have actual access to the data or perform the second computation.

An example of this is in the [JAX Documentation](https://docs.jax.dev/en/latest/501/multiprocess.html#jax-501-multiprocess). 

## This repo

The rest of this repo demonstrate a real example of the above psuedo-code and documentation. It does not require GKE or slurm or manually ssh'ing into multiple hosts. 

This example creates a 16 TPU cluster of v5e TPUs (viperlite), spanning four hosts. It uses the `worker=all` flag to the `gcloud ssh` command to enable a single command from your local machine to issue commands to the entire cluster. 

### Install latest Python and JAX

Run a command like this, with appropriate changes, from your local machine to install the latest JAX.

```sh
gcloud compute tpus tpu-vm ssh yho-v5e-16 \
    --project=tpu-pytorch \
    --zone=us-west1-c \
    --worker=all \
    --command='
      python3 -m pip install --user uv &&
      ~/.local/bin/uv venv --python 3.12 ~/venv312 &&
      ~/.local/bin/uv pip install --python ~/venv312/bin/python -U "jax[tpu]" \
        -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    '
```

### Kill existing left-over processes

Run 

```sh
./kill_all.sh
```

to cleanse any dangling processes.

### Check your setup

Run

```sh
./run.sh hello.py
```

to validate that all hosts and TPUs can successfully rendezvous.

### Execute MPMD pipeline parallel

Run

```sh
./run.sh pipeline_example.py
```

to execute a basic example. The script will `scp` the script to all four hosts and then execute it. 

###

Below are selections from the logs. You can see that in step 1a and 1b, host 0 and host 1 see actual data (adressable shards is not empty), while host 2 and 3 have handles to jax.Arrays with no addressable shards. Warning messages have been removed and the hosts are sorted, for legibility. 

```
[host 0] Total slice devices: 16, devices per stage: 8
[host 1] Total slice devices: 16, devices per stage: 8
[host 2] Total slice devices: 16, devices per stage: 8
[host 3] Total slice devices: 16, devices per stage: 8

[host 0] Step 1a (input): addressable_shards=4 data=[0.]
[host 1] Step 1a (input): addressable_shards=4 data=[2.]
[host 2] Step 1a (input): addressable_shards=0 data=none
[host 3] Step 1a (input): addressable_shards=0 data=none

[host 0] Step 1b (math op): addressable_shards=4 data=[1.]
[host 1] Step 1b (math op): addressable_shards=4 data=[3.]
[host 2] Step 1b (math op): addressable_shards=0 data=none
[host 3] Step 1b (math op): addressable_shards=0 data=none

[host 0] Step 2a (transfer): addressable_shards=0 data=none
[host 1] Step 2a (transfer): addressable_shards=0 data=none
[host 2] Step 2a (transfer): addressable_shards=4 data=[1.]
[host 3] Step 2a (transfer): addressable_shards=4 data=[3.]

[host 0] Step 2b (math op): addressable_shards=0 data=none
[host 1] Step 2b (math op): addressable_shards=0 data=none
[host 2] Step 2b (math op): addressable_shards=4 data=[2.]
[host 3] Step 2b (math op): addressable_shards=4 data=[6.]
```