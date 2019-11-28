from pathlib import Path
import torch.nn as nn
from models import BiLSTM
from datasets import TokenMapper, SupervisedDataset
from configs import BiLSTMConfig, TrainingConfig
from trainers import ModelTrainer


if __name__ == '__main__':
    # define directories for data and configs
    data_dir = Path("labeled_data")
    configs_dir = Path("config_files")

    # define paths to datasets
    train_path = data_dir / "en_ewt-ud-train.conllu"
    dev_path = data_dir / "en_ewt-ud-dev.conllu"

    # create mapper and Pytorch's dataset objects
    mapper = TokenMapper(min_frequency=5)
    mapper.create_mapping(train_path)
    train_dataset = SupervisedDataset(train_path, mapper)
    # dev_dataset = SupervisedDataset(dev_path, mapper)

    # create model
    bi_lstm_config = BiLSTMConfig().from_json_file(configs_dir / "BiLSTM_config.json")
    model = BiLSTM(bi_lstm_config, mapper)

    # create trainer object and train
    train_config = TrainingConfig().from_json_file(configs_dir / "training_config.json")
    loss_function = nn.CrossEntropyLoss(ignore_index=0)  # index 0 is used for padding
    trainer = ModelTrainer(model, train_config, loss_function)
    trainer.train(train_dataset)
