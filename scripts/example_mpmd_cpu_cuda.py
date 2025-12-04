"""Proof-of-concept that PyTorch does support limited MPMD already.

The "cluster" in question is composed of two devices: The CUDA GPU and the CPU.

We use a linear regression as the stand-in for a complex model. The
inference server runs on cuda because pytorch cuda supports async ops. We add
a cuda-sleep op to simulate a small, slow inference cluster.

The training server runs on CPU and does a much heavier weight computation.

Note that the rollout server runs at a different pace than the training server.
We find empirically that we have a replay ratio of about 9:1. Each
rollout is re-used about 9 times.

```
$ python scripts/example_mpmd_cpu_cuda.py

rollout_step=0

rollout_step=1
  train_step=0, loss.item()=0.3418

rollout_step=2
  train_step=0, loss.item()=1.2500
  train_step=1, loss.item()=0.4238
  train_step=2, loss.item()=0.2432

rollout_step=3
  train_step=0, loss.item()=0.0036
  train_step=1, loss.item()=0.0036
  train_step=2, loss.item()=0.0032
  train_step=3, loss.item()=0.0032

rollout_step=4
  train_step=0, loss.item()=0.0017
  train_step=1, loss.item()=0.0016
  train_step=2, loss.item()=0.0016
  train_step=3, loss.item()=0.0015

rollout_step=5
  train_step=0, loss.item()=0.0051
  train_step=1, loss.item()=0.0050
  train_step=2, loss.item()=0.0047
  train_step=3, loss.item()=0.0045

rollout_step=6
  train_step=0, loss.item()=0.0018
  train_step=1, loss.item()=0.0018
  train_step=2, loss.item()=0.0018
  train_step=3, loss.item()=0.0016
  train_step=4, loss.item()=0.0015

rollout_step=7
  train_step=0, loss.item()=0.0042
  train_step=1, loss.item()=0.0041
  train_step=2, loss.item()=0.0039
  train_step=3, loss.item()=0.0035

rollout_step=8
  train_step=0, loss.item()=0.0042
  train_step=1, loss.item()=0.0039
  train_step=2, loss.item()=0.0038
  train_step=3, loss.item()=0.0036

rollout_step=9
  train_step=0, loss.item()=0.0033
  train_step=1, loss.item()=0.0032
  train_step=2, loss.item()=0.0031
  train_step=3, loss.item()=0.0029
```
"""

import torch


BATCH_SIZE = 4
INPUT_DIM = 2**20
OUTPUT_DIM = 1

torch.set_default_dtype(torch.bfloat16)


def rl_inference_rollout(model, prompt):
    """Runs one step of inference on a small, slow cluster.

    Args:
        model: The model.
        prompt: The input prompt tensor.

    Returns:
        A scalar representing the "completion", on CPU
    """
    model.eval()
    rollout = model(prompt)
    # Simulate sampling by fuzzing the "rollout"
    rollout = rollout + (torch.rand_like(rollout, requires_grad=False) * 0.1)
    # Simulate a "small, slow" cluster by sleeping on CUDA
    torch.cuda._sleep(10**7)
    return rollout


def rl_training_step(model, prompt, rollout, criterion, optimizer):
    """Runs one step of training on a "larger, faster" cluster.

    Returns:
        A scalar loss.
    """
    model.train()
    prediction = model(prompt)
    loss = criterion(prediction, rollout)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    return loss


def main():
    # Create models
    model_inference = torch.nn.Linear(INPUT_DIM, OUTPUT_DIM).to("cuda")
    model_training = torch.nn.Linear(INPUT_DIM, OUTPUT_DIM).to("cpu")

    # Create optimizer and loss
    optimizer = torch.optim.SGD(model_training.parameters(), lr=1e-6)
    criterion = torch.nn.MSELoss()

    replay_buffer = None

    for rollout_step in range(10):
        print(f"\n{rollout_step=}")
        # Create random prompt
        prompt = torch.randn(BATCH_SIZE, INPUT_DIM)

        # Inference on "small, slow" cluster
        done_event = torch.cuda.Event()

        with torch.inference_mode():
            rollout = rl_inference_rollout(model_inference, prompt.to("cuda"))
            done_event.record()

        if rollout_step == 0:
            done_event.synchronize()  # Ensure first step is complete and the replay buffer has something
            replay_buffer = (prompt.cpu(), rollout.cpu())
            continue

        # On rollout_step > 0, we have a replay buffer
        # Poll until rollout is ready
        for train_step in range(100):
            if done_event.query():
                break
            loss = rl_training_step(
                model_training,
                replay_buffer[0],
                replay_buffer[1],
                criterion,
                optimizer,
            )
            print(f"  {train_step=}, {loss.item()=:.4f}")

        model_inference.load_state_dict(model_training.state_dict())
        replay_buffer = (prompt.cpu(), rollout.cpu())


if __name__ == "__main__":
    main()
