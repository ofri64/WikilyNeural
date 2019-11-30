from pathlib import Path
from models import BiLSTM
from datasets import SupervisedDataset
from configs import InferenceConfig
from trainers import ModelTrainer
from predictors import BiLSTMGreedyInferer

if __name__ == '__main__':
    # define directories for data and configs
    data_dir = Path("labeled_data")
    configs_dir = Path("config_files")
    checkpoints_dir = Path("checkpoints")

    # define paths to datasets and model checkpoint
    dev_path = data_dir / "en_ewt-ud-dev.conllu"
    config_path = configs_dir / "inference_config.json"
    checkpoint = checkpoints_dir / "BiLSTM" / "30-11-19_best_model.pth"

    # create an inference object
    trained_model: BiLSTM = ModelTrainer.load_trained_model(checkpoint, BiLSTM)
    inference_config: InferenceConfig = InferenceConfig().from_json_file(config_path)

    model_inferer = BiLSTMGreedyInferer(inference_config, trained_model)

    # create dataset and preform inference
    inference_mapper = trained_model.mapper
    dev_dataset = SupervisedDataset(dev_path, inference_mapper, sequence_len=32)

    model_inferer.infer_dataset(dev_dataset)
