import argparse
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

import wandb

from architecture import TabFound
import os
os.environ['WANDB_INIT_TIMEOUT'] = '600'


def load_batch(filename):
    f = np.load(filename)
    return f["x"], f["y"]


def main(args):
    directory = os.path.join("data", "single", "1D")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    wandb.init(
        project='FoundationOpt',
        config=args
    )

    X = []
    y = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".npz"):
            part_x, part_y = load_batch(os.path.join(directory, filename)) # f"data_{i}.npz"))
            X.append(part_x)
            y.append(part_y)

    # for i in range(0, 10):
    #     part_x, part_y = load_batch(os.path.join(data_folder, f"data_{i}.npz"))
    #     X.append(part_x)
    #     y.append(part_y)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    train_size = int(0.8 * X.shape[0])
    X_train, y_train = X[:train_size, :, :], y[:train_size, :]
    X_test, y_test = X[train_size:, :], y[train_size:, :]
    X_train = np.concatenate([X_train, y_train], axis=2)
    X_test = np.concatenate([X_test, y_test], axis=2)

    input_features = X_train.shape[2]
    model = TabFound(
        input_features=input_features,
        nr_blocks=args.nr_blocks,
        nr_heads=args.nr_heads,
        dropout=args.dropout_rate,
        nr_hyperparameters=input_features,
    )
    wandb.watch(model)
    nr_epochs = args.nr_epochs
    batch_size = args.batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = model.to(dev)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_train = X_train.to(dev)
    y_train = y_train.to(dev)

    model.train()
    # shuffle X_train indices
    indices = np.arange(X_train.shape[0])

    total_iterations = int(X_train.shape[0] / batch_size) * nr_epochs

    def warmup(current_step: int):
        return float(current_step / max(int(total_iterations / 10), 1))

    scheduler1 = LambdaLR(optimizer, lr_lambda=warmup)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=total_iterations)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[max(int(total_iterations / 10), 1)],
    )

    for i in range(0, nr_epochs):
        np.random.shuffle(indices)
        X_train = X_train[indices, :, :]
        for j in range(0, X_train.shape[0], batch_size):
            if j + batch_size > X_train.shape[0]:
                break
            optimizer.zero_grad(set_to_none=True)
            x_batch = X_train[j:j + batch_size, 0:-1, :]

            y_batch = y_train[j:j + batch_size,:]
            y_target = X_train[j:j + batch_size, 1:, :]
            y_pred = model(x_batch)
            loss = torch.nn.functional.mse_loss(y_pred, y_target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            wandb.log({"Train Loss": loss.item(), 
                       "Learning Rate": optimizer.param_groups[0]["lr"]})

    model.eval()

    with torch.no_grad():
        X_test = torch.tensor(X_test, dtype=torch.float32)
        X_test = X_test.to(dev)
        input = X_test[:, 0:-1, :]
        y_test = X_test[:, 1:, :]
        y_pred = model(input)

        loss = torch.nn.functional.mse_loss(y_pred, y_test)
        wandb.log({"Test Loss": loss.item()})

    # save the model
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save the results.',
    )
    parser.add_argument(
        '--nr_heads',
        type=int,
        default=4,
        help='Number of attention heads.',
    )
    parser.add_argument(
        '--nr_blocks',
        type=int,
        default=6,
        help='Number of transformer blocks.',
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        help='Number of transformer blocks.',
    )
    parser.add_argument(
        '--nr_epochs',
        type=int,
        default=1000,
        help='Number of training epochs.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Number of training epochs.',
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.2,
        help='The dropout rate.',
    )

    args = parser.parse_args()
    print (args)
    main(**args)
