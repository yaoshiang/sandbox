# Sandbox

This repository contains experimental code to exercise PyTorch distributed
functionality. These are generally written as functional tests with lots of
stdout, to make them conceptually easy to understand since they look like unit tests.

It also contains utilities. 

Most of the tests assume a 2x L4 machine to minimize costs. 

## Setup

### Requirements

This project uses `torchtitan`'s requirements. Install torchtitan's requirements first:

```sh
pushd ../torchtitan
pip install -r requirements.txt
pip install -r requirements-dev.txt
popd
```

Then install the local package, `sandbox`.

```sh
pip install -e .
```


## Tests

### Running Tests

The tests use PyTorch's distributed testing infrastructure. To run the distributed tests:

```bash
# Run all tests
python -m pytest tests/

# Run a specific test file
python -m pytest tests/test_distributed.py

# Run with verbose output
python -m pytest tests/ -v
```

For multi-GPU/multi-process tests, PyTorch's distributed test base will automatically spawn the required processes.

### Test Structure

- `tests/test_distributed.py` - Distributed collective operations tests

The tests use `torch.testing._internal.common_distributed.MultiProcessTestCase` which handles spawning multiple processes for distributed testing.


## Utilities

### show_collective_plan.py

Run as 

```sh
torchrun --nproc_per_node=2 scripts/example_simple_tp.py 
```

```text

[Rank 1] Dynamo Graph (Generated Code)
--------------------------------------------------------------------------------



def forward(self, L_x_ : torch.Tensor, L_self_modules_fc1_parameters_weight_ : torch.distributed.tensor.DTensor, L_self_modules_fc2_parameters_weight_ : torch.distributed.tensor.DTensor):
    l_x_ = L_x_
    l_self_modules_fc1_parameters_weight_ = L_self_modules_fc1_parameters_weight_
    l_self_modules_fc2_parameters_weight_ = L_self_modules_fc2_parameters_weight_
    input_tensor = torch__dynamo_variables_torch_prim_from_local(l_x_);  l_x_ = None
    linear = torch._C._nn.linear(input_tensor, l_self_modules_fc1_parameters_weight_, None);  input_tensor = l_self_modules_fc1_parameters_weight_ = None
    outputs = torch__dynamo_variables_tensor_prim_redistribute(linear);  linear = None
    hook_result = torch__dynamo_variables_tensor_prim_to_local(outputs);  outputs = None
    x = torch.nn.functional.relu(hook_result, inplace = False);  hook_result = None
    input_tensor_1 = torch__dynamo_variables_torch_prim_from_local_1(x);  x = None
    linear_1 = torch._C._nn.linear(input_tensor_1, l_self_modules_fc2_parameters_weight_, None);  input_tensor_1 = l_self_modules_fc2_parameters_weight_ = None
    outputs_1 = torch__dynamo_variables_tensor_prim_redistribute_1(linear_1);  linear_1 = None
    hook_result_1 = torch__dynamo_variables_tensor_prim_to_local_1(outputs_1);  outputs_1 = None
    return (hook_result_1,)

...
[Rank 1] ATen Graph (Generated Code)
--------------------------------------------------------------------------------



def forward(self, primals_1, primals_2, primals_3):
    view = torch.ops.aten.view.default(primals_1, [4, 8]);  primals_1 = None
    permute_1 = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
    mm_1 = torch.ops.aten.mm.default(view, permute_1);  permute_1 = None
    view_1 = torch.ops.aten.view.default(mm_1, [4, 16]);  mm_1 = None
    relu = torch.ops.aten.relu.default(view_1);  view_1 = None
    view_2 = torch.ops.aten.view.default(relu, [4, 16])
    permute_3 = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
    mm_3 = torch.ops.aten.mm.default(view_2, permute_3)
    all_reduce = torch.ops._c10d_functional.all_reduce.default(mm_3, 'sum', '0');  mm_3 = None
    wait_tensor = torch.ops._c10d_functional.wait_tensor.default(all_reduce);  all_reduce = None
    view_3 = torch.ops.aten.view.default(wait_tensor, [4, 8]);  wait_tensor = None
    return (view_3, view, relu, view_2, permute_3)
```
