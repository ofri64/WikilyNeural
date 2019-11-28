import torch
import torch.nn as nn
from torch.utils import data
from configs import TrainingConfig


class ModelTrainer(object):
    def __init__(self, model: nn.Module, train_config: TrainingConfig,
                 loss_function):
        self.model = model
        self.train_config = train_config
        self.loss_function = loss_function

    def train(self, train_dataset: data.Dataset, dev_dataset: data.Dataset = None):
        with_dev = dev_dataset is not None

        # training hyper parameters and configuration
        batch_size = self.train_config.batch_size
        num_workers = self.train_config.num_workers
        num_epochs = self.train_config.num_epochs
        learning_rate = self.train_config.learning_rate
        print_batch_step = self.train_config.print_step
        device = torch.device(self.train_config.device)

        # model, loss and optimizer
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # create data loaders
        train_config_dict = {"batch_size": batch_size, "num_workers": num_workers}
        training_loader = data.DataLoader(train_dataset, **train_config_dict)
        if with_dev:
            dev_loader = data.DataLoader(dev_dataset, **train_config_dict)

        # Start training
        model = model.to(device)
        model.train(mode=True)
        for epoch in range(num_epochs):

            running_loss = 0
            epoch_loss = 0

            for i, sample in enumerate(training_loader, 1):

                x, y = sample
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                output = model(x)
                loss = self.loss_function(output, y)

                loss.backward()
                optimizer.step()

                # print inter epoch statistics
                running_loss += loss.item()
                epoch_loss += loss.item() * batch_size
                if i % print_batch_step == 0:
                    print(
                        f"Epoch: {epoch + 1}, Batch Number: {i}, Average Loss: {(running_loss / print_batch_step):.3f}")
                    running_loss = 0

            epoch_num = epoch + 1
            average_epoch_loss = epoch_loss / len(train_dataset)
            print(f"Epoch {epoch_num}: average loss is {average_epoch_loss}")