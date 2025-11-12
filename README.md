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

