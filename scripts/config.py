class Config:
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "resnet50"

    TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE = "layer.3|layer.4"

    BATCH_SIZE = 32
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    REGRESSOR_LR = 5e-4
    EPOCHS = 10
    DROPOUT_PROB = 0.3
    HIDDEN_DIM = 256

    TARGET_LOSS = 50

    TRAIN_DF_PATH = "./data/train_dataset.csv"
    VAL_DF_PATH = "./data/test_dataset.csv"
    SAVE_PATH = "./models/calories_predictor.pth"


CONFIG = Config()
