import argparse
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR, ExponentialLR
from tqdm import tqdm
import wandb
import optuna
from optuna import Trial
from datetime import datetime
from model.architecture import TabFound
from datetime import datetime

os.environ['WANDB_INIT_TIMEOUT'] = '600'

def load_batch(filename):
    f = np.load(filename)
    return f["x"], f["y"]

def get_data(directory):
    X = []
    y = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".npz"):
            part_x, part_y = load_batch(os.path.join(directory, filename)) 
            X.append(part_x)
            y.append(part_y)

    bins = torch.linspace(0, 1.0, 32)
    X = np.concatenate(X, axis=0)
    X = torch.tensor(X)
    y = np.concatenate(y, axis=0)
    y = torch.tensor(y)

    bin_indices = torch.bucketize(X, bins) - 1
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Compute bin centers
    discretized_tensor_x = bin_centers[bin_indices]
    X = discretized_tensor_x

    bin_indices = torch.bucketize(y, bins) - 1
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Compute bin centers
    discretized_tensor_y = bin_centers[bin_indices]
    y = discretized_tensor_y


    # Normalize X
    #X_min = X.min(axis=0)
    #X_max = X.max(axis=0)
    #X = (X - X_min) / (X_max - X_min + 1e-10)  # Adding a small constant to avoid division by zero

    # Normalize y
    #y_min = y.min()
    #y_max = y.max()
    #y = (y - y_min) / (y_max - y_min + 1e-10)  # Adding a small constant to avoid division by zero

    return X, y 


def split_data(X, y):
    train_size = int(0.8 * X.shape[0])
    X_train, y_train = X[:train_size, :, :], y[:train_size, :]
    X_test, y_test = X[train_size:, :], y[train_size:, :]

    # Further split the training set into training and validation sets
    val_size = int(0.1 * X_train.shape[0])  # 10% of the training data for validation
    X_train, X_val = X_train[val_size:, :, :], X_train[:val_size, :, :]
    y_train, y_val = y_train[val_size:, :], y_train[:val_size, :]

    X_train = np.concatenate([X_train, y_train], axis=2)
    X_test = np.concatenate([X_test, y_test], axis=2)
    X_val = np.concatenate([X_val, y_val], axis=2)

    return X_train, X_test, X_val, y_train, y_test, y_val


def run_study(trial: Trial, X, y, args) -> float:
    formatted_datetime = datetime.now().strftime("%y%m%d")
    run = wandb.init(
        project='FoundOpt',
        config=args,
        reinit=True
    )
    
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(X, y)

    lr = trial.suggest_categorical("learning_rate", [1e-2, 1e-3, 1e-4])
    dropout_rate = trial.suggest_float("dropout_rate", 0, 0.1)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
    nr_blocks = trial.suggest_int("nr_blocks", 4, 8)
    embd_size = trial.suggest_categorical("embds_size", [32,64,128])

    input_features = X_train.shape[2]
    model = TabFound(
        input_features=input_features,
        mean_embedding_value=embd_size,
        nr_blocks=nr_blocks,
        nr_heads=args.nr_heads,
        dropout=dropout_rate,
        nr_hyperparameters=input_features,
    )

    run.watch(model)
    
    nr_epochs = args.nr_epochs
    # batch_size = args.batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = model.to(dev)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_train = X_train.to(dev)
    y_train = y_train.to(dev)

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)    
    X_val = X_val.to(dev)
    y_val = y_val.to(dev)  

    model.train()
    # shuffle X_train indices
    indices = np.arange(X_train.shape[0])

    # Define the total number of iterations (steps) per epoch
    iterations_per_epoch = int(X_train.shape[0] / batch_size)

    # Define the warm-up duration in terms of steps (independent of epochs)
    warmup_steps = 2000 # iterations_per_epoch * (0.05 * nr_epochs) # 5000  # Example: 1000 steps for warm-up
    plateau_steps = 1000 # iterations_per_epoch * (0.05 * nr_epochs) # 500  # Optional: Add a plateau phase after warm-up

    # Define the total number of iterations (steps) for the entire training
    # total_iterations = iterations_per_epoch * nr_epochs

    # Warm-up function (independent of epochs)
    def warmup_with_plateau(current_step: int):
        if current_step < warmup_steps:
            return float((current_step / warmup_steps) ** 2)  # Quadratic warm-up
        elif current_step < warmup_steps + plateau_steps:
            return 1.0  # Plateau phase
        else:
            return 0.0  # Transition to cosine annealing

    # Schedulers
    scheduler1 = LambdaLR(optimizer, lr_lambda=warmup_with_plateau)
    scheduler2 = CosineAnnealingLR(optimizer, T_max=1000) # total_iterations - warmup_steps - plateau_steps)
    scheduler3 = ExponentialLR(optimizer, gamma=0.7)
    # Sequential scheduler
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler1, scheduler2],
        milestones=[warmup_steps + plateau_steps],
    )

    #total_iterations = int(X_train.shape[0] / batch_size) * nr_epochs

    #def warmup(current_step: int):
    #    return float(current_step / max(int(total_iterations / 10), 1))

    #scheduler1 = LambdaLR(optimizer, lr_lambda=warmup)
    #scheduler2 = CosineAnnealingLR(optimizer, T_max=total_iterations)
    #scheduler = SequentialLR(
    #    optimizer,
    #    schedulers=[scheduler1, scheduler2],
    #    milestones=[max(int(total_iterations / 10), 1)],
    #)

    patience = 10
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for i in tqdm(range(0, nr_epochs), desc="Epochs"):
        np.random.shuffle(indices)
        X_train = X_train[indices, :25, :]

        # Training loop
        model.train()
        for j in tqdm(range(0, X_train.shape[0], batch_size), desc="Batches"):
            if j + batch_size > X_train.shape[0]:
                break
            optimizer.zero_grad(set_to_none=True)
            
            x_batch = X_train[j:j + batch_size, 0:-1, :]
            # y_batch = y_train[j:j + batch_size,:]
            y_target = X_train[j:j + batch_size, 1:, :]
            y_pred = model(x_batch)
            
            loss = torch.nn.functional.mse_loss(y_pred, y_target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            run.log({"Train Loss": loss.item(), 
                       "Learning Rate": optimizer.param_groups[0]["lr"]})
            
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for j in range(0, X_val.shape[0], batch_size):
                if j + batch_size > X_val.shape[0]:
                    break
                x_batch = X_val[j:j + batch_size, 0:-1, :]
                # y_batch = y_val[j:j + batch_size, :]
                y_target = X_val[j:j + batch_size, 1:, :]
                y_pred = model(x_batch)
                loss = torch.nn.functional.mse_loss(y_pred, y_target)
                val_loss += loss.item()

        val_loss /= (X_val.shape[0] / batch_size)
        run.log({"Validation Loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # save the best model
            formatted_datetime = datetime.now().strftime("%y%m%d")
            fstring = f"{args.output_dir}/output_{formatted_datetime}/trial_{trial.number}/"
            if not os.path.exists(fstring):
                os.makedirs(fstring)
                torch.save(model.state_dict(), os.path.join(fstring, "best_model.pt"))
        else:
            epochs_no_improve += 1
            print (f"Strike {epochs_no_improve} of {patience} @ epoch{i}/{nr_epochs}!")
            if epochs_no_improve == patience:
                print(f"Early stopping at epoch {i} with best validation loss: {best_val_loss}!")
                break

    print ("Training completed!")

    # Test Loop
    model.eval()

    eval_loss = 0.0
    with torch.no_grad():
        X_test= torch.tensor(X_test, dtype=torch.float32)
        X_test = X_test.to(dev)
        for j in range(0, X_test.shape[0], batch_size):
            if j + batch_size > X_test.shape[0]:
                break
            x_batch = X_test[j:j + batch_size, 0:-1, :]
            y_target = X_test[j:j + batch_size, 1:, :]
            y_pred = model(x_batch)
            eval_loss += torch.nn.functional.mse_loss(y_pred, y_target)
        run.log({"Test Loss": eval_loss.item()})
        #X_test = torch.tensor(X_test, dtype=torch.float32)
        #X_test = X_test.to(dev)
        #input = X_test[:, 0:-1, :]
        #y_test = X_test[:, 1:, :]
        #y_pred = model(input)

        #loss = torch.nn.functional.mse_loss(y_pred, y_test)
        #wandb.log({"Test Loss": loss.item()})

    # save the model
    #torch.save(model.state_dict(), os.path.join(args.output_dir, "model.pt"))
    run.unwatch()
    run.finish()
    return eval_loss

def main(args):
    # directory = os.path.join("data", "single", "1D")
    directory = os.path.join("datasets", "single", "merged")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    X, y = get_data(directory)
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: run_study(trial, X, y, args), n_trials=5, n_jobs=-1) # , callbacks=[wandbc])
    #  study.optimize(run_study, n_trials=10, n_jobs=-1) # , callbacks=[wandbc])

    print("Best trial:")
    trial = study.best_trial
    print(f"\tNumber: {study.best_trial.number}")
    print(f"\tValue: {trial.value}")
    print("\tParams: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


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
        help='Set the learning rate.',
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
        default=1024,
        help='Batch size.',
    )
    parser.add_argument(
        '--dropout_rate',
        type=float,
        default=0.2,
        help='The dropout rate.',
    )
    #global args
    args = parser.parse_args()
    main(args)
