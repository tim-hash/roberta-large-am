import torch
from transformers import RobertaTokenizer
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import pytorch_lightning as pl


from datasets import ACL2018DataModule, ACL2018Dataset
from model import EvidenceClassifier

RANDOM_SEED = 42
MAX_SEQ_LEN = 128
ROBERTA_MODEL_NAME = 'roberta-large'
N_EPOCHS = 10
BATCH_SIZE = 8

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


TRAIN_PATH = 'evidences-acl2018/train.csv'
TEST_PATH = 'evidences-acl2018/IBMDebaterEvidenceSentences/test.csv'

tokenizer = RobertaTokenizer.from_pretrained(ROBERTA_MODEL_NAME)

train_df, val_df = train_test_split(pd.read_csv(TRAIN_PATH), test_size=0.1, random_state=RANDOM_SEED)
test_df = pd.read_csv(TEST_PATH)
train_df, val_df, test_df = train_df.sample(frac=1), val_df.sample(frac=1), test_df.sample(frac=1)

train_dataset = ACL2018Dataset(train_df, tokenizer, MAX_SEQ_LEN)
data_module = ACL2018DataModule(train_df, val_df, test_df, tokenizer, batch_size=BATCH_SIZE)
data_module.setup()

model = EvidenceClassifier(n_classes=1, 
                           steps_per_epochs=len(train_dataset)//BATCH_SIZE, 
                           n_epochs=N_EPOCHS
                           )

tb_logger = pl.loggers.TensorBoardLogger('logs/')

trainer = pl.Trainer(max_epochs=N_EPOCHS, gpus=2, progress_bar_refresh_rate=30, 
                     logger=tb_logger, accelerator='ddp')

torch.cuda.empty_cache()
trainer.fit(model, data_module)
trainer.save_checkpoint('checkpoint-model-roberta-large.ckpt')