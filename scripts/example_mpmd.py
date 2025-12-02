"""Proof-of-concept that PyTorch does support MPMD already: cuda and cpu.

We use a linear regression as the stand-in for a complex model. The
inference server runs on cuda because pytorch cuda supports async ops. We add
a cuda-sleep op to simulate a small, slow inference cluster.

The training server runs on CPU and does a much heavier weight computation.

```
(tt) yho_google_com@yho-l4:~/Documents/GitHub/sandbox$ python scripts/example_mpmd.py 

rollout_step=0

rollout_step=1
  train_step=0, loss.item()=0.7277
  train_step=1, loss.item()=0.7257
  train_step=2, loss.item()=0.7304

rollout_step=2
  train_step=0, loss.item()=0.0471
  train_step=1, loss.item()=0.0534
  train_step=2, loss.item()=0.0444
  train_step=3, loss.item()=0.0415
  train_step=4, loss.item()=0.0517
  train_step=5, loss.item()=0.0443
  train_step=6, loss.item()=0.0531
  train_step=7, loss.item()=0.0510
  train_step=8, loss.item()=0.0405

rollout_step=3
  train_step=0, loss.item()=0.0489
  train_step=1, loss.item()=0.0475
  train_step=2, loss.item()=0.0567
  train_step=3, loss.item()=0.0472
  train_step=4, loss.item()=0.0443
  train_step=5, loss.item()=0.0552
  train_step=6, loss.item()=0.0489
  train_step=7, loss.item()=0.0517
  train_step=8, loss.item()=0.0398

rollout_step=4
  train_step=0, loss.item()=0.0482
  train_step=1, loss.item()=0.0457
  train_step=2, loss.item()=0.0448
  train_step=3, loss.item()=0.0467
  train_step=4, loss.item()=0.0543
  train_step=5, loss.item()=0.0547
  train_step=6, loss.item()=0.0488
  train_step=7, loss.item()=0.0484
  train_step=8, loss.item()=0.0431

rollout_step=5
  train_step=0, loss.item()=0.0520
  train_step=1, loss.item()=0.0539
  train_step=2, loss.item()=0.0551
  train_step=3, loss.item()=0.0536
  train_step=4, loss.item()=0.0448
  train_step=5, loss.item()=0.0561
  train_step=6, loss.item()=0.0429
  train_step=7, loss.item()=0.0489
  train_step=8, loss.item()=0.0562

rollout_step=6
  train_step=0, loss.item()=0.0512
  train_step=1, loss.item()=0.0381
  train_step=2, loss.item()=0.0466
  train_step=3, loss.item()=0.0413
  train_step=4, loss.item()=0.0475
  train_step=5, loss.item()=0.0379
  train_step=6, loss.item()=0.0540
  train_step=7, loss.item()=0.0435
  train_step=8, loss.item()=0.0530

rollout_step=7
  train_step=0, loss.item()=0.0501
  train_step=1, loss.item()=0.0434
  train_step=2, loss.item()=0.0464
  train_step=3, loss.item()=0.0473
  train_step=4, loss.item()=0.0478
  train_step=5, loss.item()=0.0422
  train_step=6, loss.item()=0.0592
  train_step=7, loss.item()=0.0468
  train_step=8, loss.item()=0.0491

rollout_step=8
  train_step=0, loss.item()=0.0444
  train_step=1, loss.item()=0.0514
  train_step=2, loss.item()=0.0518
  train_step=3, loss.item()=0.0490
  train_step=4, loss.item()=0.0563
  train_step=5, loss.item()=0.0561
  train_step=6, loss.item()=0.0451
  train_step=7, loss.item()=0.0514
  train_step=8, loss.item()=0.0544

rollout_step=9
  train_step=0, loss.item()=0.0526
  train_step=1, loss.item()=0.0474
  train_step=2, loss.item()=0.0436
  train_step=3, loss.item()=0.0475
  train_step=4, loss.item()=0.0532
  train_step=5, loss.item()=0.0484
  train_step=6, loss.item()=0.0527
  train_step=7, loss.item()=0.0582
  train_step=8, loss.item()=0.0603
```
"""

import torch


BATCH_SIZE = 32
INPUT_DIM = 1024
OUTPUT_DIM = 1

torch.set_default_dtype(torch.float64)

SLOW_CLUSTER = "cuda"
FAST_CLUSTER = "cpu"


def rl_inference_rollout(model, prompt):
    """Runs one step of inference on a small, slow cluster.

    Args:
        model: The model.
        prompt: The input prompt tensor.

    Returns:
        A scalar representing the "completion", on CPU
    """
    rollout = model(prompt)
    return rollout


def rl_training_step(model, prompt, rollout, criterion, optimizer):
    """Runs one step of training on a "larger, faster" cluster.

    Returns:
        A scalar loss.
    """
    model.eval()
    rollout = rollout
    prediction = model(prompt)
    loss = criterion(prediction, rollout)
    # Simulate a RL loss by fuzzing the loss
    loss = loss + (torch.rand_like(loss, requires_grad=False) * 0.1)
    loss = loss.mean()
    loss.backward()

    optimizer.zero_grad()
    optimizer.step()
    return loss


def main():
    # Create models
    model_inference = torch.nn.Linear(INPUT_DIM, OUTPUT_DIM).to("cuda")
    model_training = torch.nn.Linear(INPUT_DIM, OUTPUT_DIM).to("cpu")

    # Create optimizer and loss
    optimizer = torch.optim.SGD(model_training.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss(reduction="none")

    replay_buffer = None

    for rollout_step in range(10):
        print(f"\n{rollout_step=}")
        # Create random prompt
        prompt = torch.randn(BATCH_SIZE, INPUT_DIM)

        # Inference on "small, slow" cluster
        done_event = torch.cuda.Event()

        with torch.inference_mode():
            rollout = rl_inference_rollout(model_inference, prompt.to("cuda"))
            torch.cuda._sleep(10**7)
            done_event.record()

        if rollout_step == 0:
            done_event.synchronize()  # Ensure first step is complete and the replay buffer has something

        # Poll until rollout is ready
        for train_step in range(50):
            if done_event.query():
                # Always breaks on rollout_step==0
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
