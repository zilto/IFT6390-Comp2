import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

import matplotlib.pyplot as plt

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()   

        self.layers4 = nn.Sequential(
            nn.Conv1d(in_channels=18, out_channels=18, kernel_size=4),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=18),
            #nn.InstanceNorm1d(num_features=18),
            nn.Flatten(),
            #nn.Linear(in_features=162, out_features=162),
            #nn.ReLU(),
            #nn.Dropout(0),
            #nn.Linear(in_features=162, out_features=2),
        )
        
        self.layers3 = nn.Sequential(
            nn.Conv1d(in_channels=18, out_channels=18, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=18),
            #nn.InstanceNorm1d(num_features=18),
            nn.Flatten(),
            #nn.Linear(in_features=180, out_features=180),
            #nn.ReLU(),
            #nn.Dropout(0),
            #nn.Linear(in_features=180, out_features=2),
        )
    
    def forward(self, x):
        x3 = self.layers3(x)
        x4 = self.layers4(x)
        
        concat = torch.cat((x3, x4), dim=1)
        
        out_block = nn.Sequential(
            nn.Linear(in_features=342, out_features=342),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(in_features=342, out_features=2)
        )
        #x = self.layers(x)
        return out_block(concat)
        

class TrainingHarness(nn.Module):
    def __init__(
        self,
        dataset,
        model,
        lr=0.0001,
        batch_size=128,
        normalize=None,
    ):       
        super().__init__()
        
        self.train, self.valid = dataset
        if normalize:
            self.train, self.valid = self.normalize(dataset, orientation=normalize)
            
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.lr = lr
        self.batch_size = batch_size
            
        self.train_logs = {
            'accuracy': {
                "train": [],
                "validation": []
                },
            'loss': {
                "train": [],
                "validation": []
                },
            'gradient_norm': {
                "train": []
            }
        }
                

    @staticmethod
    def compute_gradient_norm(model: torch.nn.Module) -> float:
        # Compute the Euclidean norm of the gradients of the parameters of the network
        # with respect to the loss function.
        # similar to first function "gradient_norm"
        params = [p.grad.flatten() for p in model.parameters()]
        concat = torch.cat(params)
        grad_norm = concat.norm().item()
        return grad_norm
    
    def compute_loss_and_accuracy(self, X: torch.Tensor, y: torch.Tensor):
        # y is one-hot encoded
        output = self.model(X)
        y_oh = torch.nn.functional.one_hot(y.to(torch.int64))
        loss = F.binary_cross_entropy_with_logits(output, y_oh.type_as(output))

        # accuracy
        n_correct = torch.sum(output.argmax(axis=1) == y)
        accuracy = n_correct / y.shape[0]
        return (loss, accuracy.item())
    
    def training_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        # y_batch is one-hot encoded
        self.model.train() # set to Pytorch training mode to collect gradients
        
        # compute loss and accuracy (predictions are made within the function)
        loss, accuracy = self.compute_loss_and_accuracy(X_batch, y_batch)
        # backward propagate the loss
        loss.backward()
        # compute the gradient norm of the network after backpropagation
        batch_grad = self.compute_gradient_norm(self.model)
        # take a optimizer step
        self.optimizer.step()
        # reset gradient for next step
        self.optimizer.zero_grad()
        # return the gradient norm of the batch for this training step
        return batch_grad


    def log_metrics(self, X_train: torch.Tensor, y_train: torch.Tensor,
                    X_valid: torch.Tensor, y_valid: torch.Tensor) -> None:
        self.model.eval()
        with torch.inference_mode():
            train_loss, train_accuracy = self.compute_loss_and_accuracy(X_train, y_train)
            valid_loss, valid_accuracy = self.compute_loss_and_accuracy(X_valid, y_valid)
        self.train_logs["accuracy"]["train"].append(train_accuracy)
        self.train_logs["accuracy"]["validation"].append(valid_accuracy)
        self.train_logs["loss"]["train"].append(train_loss.item())
        self.train_logs["loss"]["validation"].append(valid_loss.item())
    
    
    def train_loop(self, n_epochs: int):
        # Prepare train and validation data
        X_train, y_train = self.train
        X_valid, y_valid = self.valid

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        self.log_metrics(X_train, y_train, X_valid, y_valid)
        for epoch in tqdm.tqdm(range(n_epochs)):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_train[self.batch_size * batch:self.batch_size * (batch + 1)]
                gradient_norm = self.training_step(minibatchX, minibatchY)
            # Just log the last gradient norm
            self.train_logs['gradient_norm']["train"].append(gradient_norm)
            self.log_metrics(X_train, y_train, X_valid, y_valid)
            
        return self.train_logs

    def evaluate(self, X: torch.Tensor, y: torch.Tensor):
        # during evaluation you don't want grad to be accumulated
        with torch.no_grad():
            # compute the accuracy and loss on the X, y tensors
            loss, accuracy = self.compute_loss_and_accuracy(X, y)
        return (loss, accuracy)
    
    def plot_logs(self):
        for metric in self.train_logs.keys():
            plt.figure()
            plt.title(metric)
            for line in self.train_logs[metric].keys():
                plt.plot(self.train_logs[metric][line], label=f"{line}")
            plt.legend()
            plt.show()