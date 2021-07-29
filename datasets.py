import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import pandas as pd

BATCH_SIZE = 8
MAX_SEQ_LEN = 128

class ACL2018Dataset(Dataset):
  def __init__(self, data:pd.DataFrame, tokenizer:RobertaTokenizer, max_token_len:int = 128):
    self.data = data
    self.tokenizer = tokenizer
    self.max_token_len=max_token_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index:int):
    row = self.data.iloc[index]
    topic = row['topic']
    candidate = row['candidate']
    label = row['label']

    #Might be tokenizer/model specific
    encoding = self.tokenizer.encode_plus(
    topic,
    candidate,
    add_special_tokens = True,
    max_length = self.max_token_len,
    return_token_type_ids=False,
    padding='max_length',
    truncation=True,
    return_attention_mask = True,
    return_tensors='pt'
    )
    
    return dict(
        topic=topic,
        candidate=candidate,
        input_ids = encoding['input_ids'].flatten(),
        #token_type_ids = encoding['token_type_ids'].flatten(),
        attention_mask = encoding['attention_mask'].flatten(),
        label = torch.Tensor([label])
    )


class ACL2018DataModule(pl.LightningDataModule):
  def __init__(self, train_df, val_df, test_df, tokenizer, batch_size =BATCH_SIZE, max_token_len=MAX_SEQ_LEN):
    super().__init__()
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.batch_size = batch_size
    self.max_token_len = max_token_len

  def setup(self):
    self.train_dataset = ACL2018Dataset(
        self.train_df,
        self.tokenizer,
        self.max_token_len,
    )
    self.validation_dataset = ACL2018Dataset(
        self.val_df,
        self.tokenizer,
        self.max_token_len,   
    )
    self.test_dataset = ACL2018Dataset(
        self.test_df,
        self.tokenizer,
        self.max_token_len,   
    )

  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size = self.batch_size,
        shuffle = True,
        num_workers = 32
    )
  
  def val_dataloader(self):
    return DataLoader(self.validation_dataset,batch_size = 1, num_workers = 2)

  def test_dataloader(self):
    return DataLoader(self.test_dataset,batch_size = 1, num_workers = 2)