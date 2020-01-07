import torch.nn as nn
import torch.utils.data as data

from models import BaseModel, BasicBiLSTM
from mappers import BaseMapper, BaseMapperWithPadding, TokenMapperUnkCategoryWithPadding
from predictors import BasePredictor, GreedyLSTMPredictor
from configs import BaseConfig, ModelConfig, TrainingConfig, InferenceConfig, RNNConfig
from datasets import BiLSTMDataset
from trainers import ModelTrainer


class ConfigsFactory(object):
    def __call__(self, config_type: str) -> BaseConfig:

        if config_type == "training":
            return TrainingConfig()

        if config_type == "inference":
            return InferenceConfig()

        if "lstm" in config_type:
            return RNNConfig()


class MappersFactory(object):
    def __call__(self, config: BaseConfig, mapper_name: str) -> BaseMapper:

        # mapper_attributes
        if "min_frequency" in config:
            min_frequency = config["min_frequency"]
        else:
            min_frequency = 0

        if "split_char" in config:
            split_char = config["split_char"]

        else:
            split_char = "\t"

        if "lstm" in mapper_name:
            return TokenMapperUnkCategoryWithPadding(min_frequency, split_char)


class ModelsFactory(object):

    def __call__(self, parameters_dict: BaseConfig, model_config: ModelConfig, mapper: BaseMapper, model_name: str) -> BaseModel:

        if "lstm" in model_name:
            model_config: RNNConfig
            mapper: BaseMapperWithPadding
            return BasicBiLSTM(model_config, mapper)


class PredictorsFactory(object):
    def __call__(self, parameters_dict: BaseConfig, mapper: BaseMapper, predictor_type: str) -> BasePredictor:

        if "lstm" in predictor_type:
            mapper: BaseMapperWithPadding
            return GreedyLSTMPredictor(mapper)


class DatasetsFactory(object):
    def __call__(self, config: BaseConfig, file_path: str, mapper: BaseMapper, dataset_type: str) -> data.Dataset:

        if "lstm" in dataset_type:

            if "sequence_length" in config:
                sequence_length = config["sequence_length"]
            else:
                sequence_length = 50  # default value

            mapper: BaseMapperWithPadding
            return BiLSTMDataset(file_path, mapper, sequence_length)


class TrainerFactory(object):
    def __call__(self, model: BaseModel, train_config: TrainingConfig,
                 predictor: BasePredictor, loss_function: nn.Module,
                 model_type: str) -> ModelTrainer:

        if "lstm" in model_type:
            return ModelTrainer(model, train_config, predictor, loss_function)


class LossFunctionFactory(object):
    def __call__(self, model_type: str, mapper: BaseMapper = None) -> nn.Module:

        if "lstm" in model_type:
            mapper: BaseMapperWithPadding
            padding_symbol = mapper.get_padding_symbol()
            label_padding_index = mapper.get_label_idx(padding_symbol)
            return nn.CrossEntropyLoss(ignore_index=label_padding_index)
