"""Distributed tests using PyTorch's distributed testing infrastructure."""

import copy
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed._tensor import (
    DTensor,
    Replicate,
    distribute_tensor,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.testing._internal.common_distributed import (
    DistributedTestBase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests

from .fixtures import SimpleMLP


class CollectivesTest(DistributedTestBase):
    """Tests for distributed collective operations.

    This test class uses DistributedTestBase which automatically:
    - Spawns multiple processes (defined by world_size)
    - Initializes the process group with the correct backend
    - Sets up devices correctly
    - Cleans up the process group after tests
    """

    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    def test_sanity(self):
        """Sanity check to ensure distributed setup is correct."""
        self.create_pg("cuda:0")
        self.assertEqual(dist.get_world_size(), self.world_size)
        self.assertEqual(dist.get_rank(), self.rank)

    @skip_if_lt_x_gpu(2)
    def test_all_gather(self):
        """Test all_gather operation with different tensors on each rank.

        Rank 0 has a tensor of 100, rank 1 has a tensor of 101.
        After all_gather, both ranks should have both tensors.
        """
        # Arrange
        self.create_pg("cuda:0")
        device = f"cuda:{self.rank}"
        dtype = torch.bfloat16

        tensor = (100 + self.rank) * torch.ones(2, 2, device=device, dtype=dtype)

        # Act
        gathered_tensors = [
            torch.empty(2, 2, device=device, dtype=dtype)
            for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_tensors, tensor)

        # Assert
        actual = gathered_tensors[0]
        expected = 100 * torch.ones(2, 2, device=device, dtype=dtype)
        torch.testing.assert_close(actual, expected)

        actual = gathered_tensors[1]
        expected = 101 * torch.ones(2, 2, device=device, dtype=dtype)
        torch.testing.assert_close(actual, expected)

    @skip_if_lt_x_gpu(2)
    def test_all_reduce(self):
        """Test all_reduce operation with AVG."""
        # Arrange
        self.create_pg("cuda:0")
        device = f"cuda:{self.rank}"
        dtype = torch.bfloat16

        x = self.rank * torch.ones(2, 2, device=device, dtype=dtype)

        # Act
        dist.all_reduce(x, op=dist.ReduceOp.AVG)

        # Assert
        expected = 0.5 * torch.ones(2, 2, device=device, dtype=dtype)
        torch.testing.assert_close(x, expected)

    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter(self):
        """Test reduce_scatter operation.

        Reduces a list of tensors and scatters the result to each rank.
        Each rank gets one chunk of the reduced result.
        """
        # Arrange
        self.create_pg("cuda:0")
        device = f"cuda:{self.rank}"
        dtype = torch.bfloat16

        # Rank 0: [0, 1], Rank 1: [2, 3]
        x = self.rank * self.world_size + torch.arange(2, device=device, dtype=dtype)
        expected = [2] if self.rank == 0 else [4]
        expected = torch.tensor(expected, device=device, dtype=dtype)

        # Act
        output = torch.empty(1, device=device, dtype=dtype)
        dist.reduce_scatter(output, [x[0:1], x[1:2]], op=dist.ReduceOp.SUM)

        # Assert
        torch.testing.assert_close(output, expected)

    @skip_if_lt_x_gpu(2)
    def test_all_to_all(self):
        """Test all_to_all operation."""
        # Arrange
        self.create_pg("cuda:0")
        device = f"cuda:{self.rank}"
        dtype = torch.bfloat16

        output_list = [
            torch.empty(2, 2, device=device, dtype=dtype)
            for _ in range(self.world_size)
        ]

        # Rank 0 sends: [10, 11] to ranks [0, 1]
        # Rank 1 sends: [20, 21] to ranks [0, 1]
        input_list = [
            (self.rank * 10 + i) * torch.ones(2, 2, device=device, dtype=dtype)
            for i in range(self.world_size)
        ]

        # All-to-all exchange
        dist.all_to_all(output_list, input_list)

        # Rank 0 receives: [10 from rank 0, 20 from rank 1]
        # Rank 1 receives: [11 from rank 0, 21 from rank 1]
        for i in range(self.world_size):
            expected_value = i * 10 + self.rank
            expected = expected_value * torch.ones(2, 2, device=device, dtype=dtype)
            torch.testing.assert_close(
                output_list[i],
                expected,
                msg=f"Rank {self.rank}: Expected tensor from rank {i}",
            )

    @skip_if_lt_x_gpu(2)
    def test_broadcast(self):
        """Test broadcast operation.

        Rank 0 broadcasts its tensor to all other ranks.
        """
        # Arrange
        self.create_pg("cuda:0")
        device = f"cuda:{self.rank}"
        dtype = torch.bfloat16

        if self.rank == 0:
            tensor = 42 * torch.ones(2, 2, device=device, dtype=dtype)
        else:
            tensor = torch.empty(2, 2, device=device, dtype=dtype)

        # Act
        dist.broadcast(tensor, src=0)

        # After broadcast, all ranks should have rank 0's value
        expected = 42 * torch.ones(2, 2, device=device, dtype=dtype)
        torch.testing.assert_close(tensor, expected)

    @skip_if_lt_x_gpu(2)
    def test_send_recv(self):
        """Test point-to-point send/recv operations.

        Rank 0 sends a tensor to rank 1.
        Note: This breaks SPMD pattern as ranks execute different code paths.
        """
        # Arrange
        self.create_pg("cuda:0")
        device = f"cuda:{self.rank}"
        dtype = torch.bfloat16

        if self.rank == 0:
            # Rank 0: sender
            tensor = 99 * torch.ones(2, 2, device=device, dtype=dtype)
            dist.send(tensor, dst=1)
            # Verify we still have our original value
            torch.testing.assert_close(
                tensor, 99 * torch.ones(2, 2, device=device, dtype=dtype)
            )
        elif self.rank == 1:
            # Rank 1: receiver
            tensor = torch.empty(2, 2, device=device, dtype=dtype)
            dist.recv(tensor, src=0)
            # Verify we received the correct value
            torch.testing.assert_close(
                tensor, 99 * torch.ones(2, 2, device=device, dtype=dtype)
            )
        else:
            raise RuntimeError()


class DTensorTest(DistributedTestBase):
    """Tests for DTensor (distributed tensor) operations.

    DTensor provides a higher-level API for distributed tensors with
    automatic sharding and replication strategies.
    """

    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    def test_dtensor_replicate(self):
        """Test DTensor with Replicate placement using rank 0 broadcast.

        Only rank 0 creates the actual data. distribute_tensor with Replicate
        placement broadcasts rank 0's data to all other ranks.
        """
        # Arrange
        self.create_pg("cuda:0")
        device = f"cuda:{self.rank}"
        dtype = torch.bfloat16
        device_mesh = init_device_mesh("cuda", (self.world_size,))

        # Rank 0 creates the actual tensor, the others will get replicated into.
        if self.rank == 0:
            tensor = torch.arange(1024, device=device, dtype=dtype).view(32, 32)
        else:
            tensor = torch.empty(32, 32, device=device, dtype=dtype)

        # Act. Broadcast from rank 0.
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])
        result = dtensor + 1

        # Assert: all ranks should see the updated data.
        expected_local = torch.arange(1024, device=device, dtype=dtype).view(32, 32) + 1
        torch.testing.assert_close(result.to_local(), expected_local)

    @skip_if_lt_x_gpu(2)
    def test_dtensor_replicate_overwrites_non_rank0(self):
        """Test that Replicate placement overwrites non-rank0 data with rank 0's data.

        Each rank creates different local data, but distribute_tensor with Replicate
        always uses rank 0's data and overwrites all other ranks' tensors.
        """
        # Arrange
        self.create_pg("cuda:0")
        device = f"cuda:{self.rank}"
        dtype = torch.bfloat16

        device_mesh = init_device_mesh("cuda", (self.world_size,))

        # Each rank creates DIFFERENT data
        # Rank 0: [0, 1, 2, ..., 15]
        # Rank 1: [100, 101, 102, ..., 115]
        tensor = torch.arange(16, device=device, dtype=dtype).view(4, 4) + (
            self.rank * 100
        )

        # Act - distribute_tensor with Replicate uses rank 0's data
        # This will OVERWRITE rank 1's tensor with rank 0's data
        dtensor = distribute_tensor(tensor, device_mesh, [Replicate()])

        # Assert
        # ALL ranks should now have rank 0's data (0-15), not their original data
        expected_local = torch.arange(16, device=device, dtype=dtype).view(4, 4)
        torch.testing.assert_close(dtensor.to_local(), expected_local)

        # Rank 1's original data (100-115) has been overwritten
        if self.rank == 1:
            # Verify that rank 1's original tensor is NOT preserved
            original_rank1_data = (
                torch.arange(16, device=device, dtype=dtype).view(4, 4) + 100
            )
            # This should NOT match - rank 1's data was overwritten
            with self.assertRaises(AssertionError):
                torch.testing.assert_close(dtensor.to_local(), original_rank1_data)

    @skip_if_lt_x_gpu(2)
    def test_dtensor_from_local_with_replicate(self):
        """Test DTensor.from_local with Replicate when ranks have different data.

        Unlike distribute_tensor, from_local does NOT broadcast or synchronize data.
        It trusts that you're telling the truth about the placement.
        If you say Replicate but ranks have different data, you get undefined behavior.
        """
        # Arrange
        self.create_pg("cuda:0")
        device = f"cuda:{self.rank}"
        dtype = torch.bfloat16

        device_mesh = init_device_mesh("cuda", (self.world_size,))

        # Each rank creates DIFFERENT local data
        # Rank 0: [0, 1, 2, ..., 15]
        # Rank 1: [100, 101, 102, ..., 115]
        local_tensor = torch.arange(16, device=device, dtype=dtype).view(4, 4) + (
            self.rank * 100
        )

        # Act - from_local with Replicate does NOT synchronize!
        # It just wraps the local tensor and ASSUMES all ranks have the same data
        dtensor = DTensor.from_local(local_tensor, device_mesh, [Replicate()])

        # Assert
        # Each rank STILL HAS ITS OWN DATA - no broadcast happened!
        # Rank 0 has [0-15], Rank 1 has [100-115]
        expected_local = torch.arange(16, device=device, dtype=dtype).view(4, 4) + (
            self.rank * 100
        )
        torch.testing.assert_close(dtensor.to_local(), expected_local)

        # This is dangerous! The DTensor claims to be Replicate but isn't actually replicated
        # Operations on this DTensor may produce incorrect results because PyTorch
        # assumes all ranks have identical data
        self.assertEqual(dtensor.placements, (Replicate(),))


class ShardingTest(DistributedTestBase):
    """Tests for sharding for models."""

    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    def test_tp_weight_shapes(self):
        """Verify TP shards weights to correct local shapes."""
        # Arrange environment
        self.create_pg("cuda:0")
        device = f"cuda:{self.rank}"
        dtype = torch.bfloat16
        device_mesh = init_device_mesh("cuda", (self.world_size,))

        # Arrange expected value
        torch.manual_seed(123)
        model = SimpleMLP(8, 32, 8).to(device=device, dtype=dtype)
        parallelize_module(
            model,
            device_mesh,
            {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()},
        )

        # Verify sharded shapes. The order of dimenions gets transposed
        # but the point is that each rank outputs half the 32 values on
        # col parallel, and row parallel does a partial matmul.
        self.assertEqual(model.fc1.weight.to_local().shape, (16, 8))
        self.assertEqual(model.fc2.weight.to_local().shape, (8, 16))

        # Verify ranks have different weight data
        gathered = [
            torch.empty(16, 8, device=device, dtype=dtype)
            for _ in range(self.world_size)
        ]
        dist.all_gather(gathered, model.fc1.weight.to_local())
        if self.rank == 0:
            self.assertFalse(torch.allclose(gathered[0], gathered[1]))
            torch.testing.assert_close(gathered[0], model.fc1.weight.to_local())
        elif self.rank == 1:
            torch.testing.assert_close(gathered[1], model.fc1.weight.to_local())

    @skip_if_lt_x_gpu(2)
    def test_mlp_replicated_vs_tp_sharded_has_same_output(self):
        """Test TP-sharded MLP produces same output as replicated MLP."""
        # Arrange
        self.create_pg("cuda:0")
        device = f"cuda:{self.rank}"
        dtype = torch.bfloat16
        device_mesh = init_device_mesh("cuda", (self.world_size,))

        # Expected. Generate expected by running model locally.
        torch.manual_seed(42)
        input_tensor = torch.randn(3, 8, device=device, dtype=dtype)
        model = SimpleMLP(8, 16, 8).to(device=device, dtype=dtype)
        expected_output = model(input_tensor)

        # Act. Create sharded model.
        sharded_model = copy.deepcopy(model)
        parallelize_module(
            sharded_model,
            device_mesh,
            {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()},
        )
        sharded_output = sharded_model(input_tensor)

        # Assert
        torch.testing.assert_close(
            sharded_output, expected_output, rtol=1e-2, atol=1e-2
        )


class DCPTest(DistributedTestBase):
    """Tests for Distributed Checkpoint (DCP) with sharded models."""

    @property
    def world_size(self) -> int:
        return 2

    @skip_if_lt_x_gpu(2)
    def test_dcp_with_tensor_parallel(self):
        """Test DCP save/load with TP-sharded model."""
        self.create_pg("cuda:0")
        device = f"cuda:{self.rank}"
        dtype = torch.bfloat16
        device_mesh = init_device_mesh("cuda", (self.world_size,))

        # Create and shard model
        torch.manual_seed(123)
        model = SimpleMLP(8, 16, 8).to(device=device, dtype=dtype)
        parallelize_module(
            model,
            device_mesh,
            {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()},
        )

        # Save checkpoint
        state_dict = model.state_dict()
        checkpoint_dir = tempfile.mkdtemp() if self.rank == 0 else None
        checkpoint_dir_list = [checkpoint_dir]
        dist.broadcast_object_list(checkpoint_dir_list, src=0)
        checkpoint_path = Path(checkpoint_dir_list[0])

        try:
            dcp.save(state_dict=state_dict, checkpoint_id=checkpoint_path)
            dist.barrier()

            # Load checkpoint into new model
            torch.manual_seed(456)
            model_loaded = SimpleMLP(8, 16, 8).to(device=device, dtype=dtype)
            parallelize_module(
                model_loaded,
                device_mesh,
                {"fc1": ColwiseParallel(), "fc2": RowwiseParallel()},
            )

            loaded_state_dict = model_loaded.state_dict()
            dcp.load(state_dict=loaded_state_dict, checkpoint_id=checkpoint_path)
            model_loaded.load_state_dict(loaded_state_dict)

            # Verify loaded weights match original
            torch.testing.assert_close(
                model_loaded.fc1.weight.to_local(),
                state_dict["fc1.weight"].to_local(),
                rtol=1e-5,
                atol=1e-5,
            )
            torch.testing.assert_close(
                model_loaded.fc2.weight.to_local(),
                state_dict["fc2.weight"].to_local(),
                rtol=1e-5,
                atol=1e-5,
            )

        finally:
            if self.rank == 0:
                import shutil

                shutil.rmtree(checkpoint_path, ignore_errors=True)


if __name__ == "__main__":
    run_tests()
