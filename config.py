import pandas as pd

# PATH
DIR_TRAIN_BEN = "/scratch.global/yin00406/Cashew_WA/P4B/TRAIN/BEN"
DIR_TRAIN_CIV = "/scratch.global/yin00406/Cashew_WA/P4B/TRAIN/CIV"
DIR_TRAIN_SEN = "/scratch.global/yin00406/Cashew_WA/SEN/P4B/TRAIN"
DIR_TRAIN_GHA = "/scratch.global/yin00406/Cashew_WA/GHA/P4B/TRAIN"

# IMAGERY INFO
STEP = [202012, 202102, 202103, 202104]
STEP_2023 = [202312, 202401, 202402, 202403, 202404]
STEP_2018 = [201712, 201806]
BAND_NUM = 4
CLS_NUM = 3
PATCH_SIZE = 64

CSV_TRAIN_BEN = "/users/6/yin00406/Cashew_mapping_WA/072024/TRAIN/BEN.csv"
TILES_TRAIN_BEN = pd.read_csv(CSV_TRAIN_BEN, header=None).values.tolist()
TILES_TRAIN_BEN = [item for sublist in TILES_TRAIN_BEN for item in sublist]

CSV_TRAIN_CIV = "/users/6/yin00406/Cashew_mapping_WA/072024/TRAIN/CIV_TRAIN.csv"
TILES_TRAIN_CIV = pd.read_csv(CSV_TRAIN_CIV, header=None).values.tolist()
TILES_TRAIN_CIV = [item for sublist in TILES_TRAIN_CIV for item in sublist]

CSV_TRAIN_SEN = "/users/6/yin00406/Cashew_mapping_WA/072024/TRAIN/SEN_DA.csv"
TILES_TRAIN_SEN = pd.read_csv(CSV_TRAIN_SEN, header=None).values.tolist()
TILES_TRAIN_SEN = [item for sublist in TILES_TRAIN_SEN for item in sublist]

CSV_TRAIN_GHA = "/users/6/yin00406/Cashew_mapping_WA/072024/TRAIN/GHA.csv"
TILES_TRAIN_GHA = pd.read_csv(CSV_TRAIN_GHA, header=None).values.tolist()
TILES_TRAIN_GHA = [item for sublist in TILES_TRAIN_GHA for item in sublist]

n_epochs = 500
n_epochs_DA = 500
unknown_class = 10
iterations = 1
small_object_threshold = 5