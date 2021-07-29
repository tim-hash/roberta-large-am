from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import torch

ROBERTA_MODEL_NAME = 'roberta-large'

class EvidenceClassifier(pl.LightningModule):
  def __init__(self, n_classes:int, steps_per_epochs=None, n_epochs=None):
    super().__init__()
    self.roberta = RobertaModel.from_pretrained(ROBERTA_MODEL_NAME, return_dict=True)
    self.classifier = nn.Linear(self.roberta.config.hidden_size, n_classes)
    self.steps_per_epochs= steps_per_epochs
    self.n_epochs = n_epochs
    self.criterion = nn.BCELoss()
    self.accuracy = torchmetrics.Accuracy(threshold=0.4)
  
  def forward(self, input_ids, attention_mask, label=None):
    output = self.roberta(input_ids, attention_mask=attention_mask)
    output = self.classifier(output.pooler_output)
    output = torch.sigmoid(output)
    loss = 0
    if label is not None:
      loss = self.criterion(output, label)
    return loss, output

  def training_step(self, batch, batch_idx):
    input_ids =batch['input_ids']
    attention_mask = batch['attention_mask']
    label = batch['label']
    loss , output = self(input_ids, attention_mask, label)
    self.log('train loss', loss, prog_bar=True, logger=True)
    self.log('train acc', self.accuracy(output, label.int()))
    return {'loss': loss, 'predictions':output, 'label':label}
  
  def validation_step(self, batch, batch_idx):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    label = batch['label']
    loss , output = self(input_ids, attention_mask, label)
    self.log('val loss', loss, prog_bar=True, logger=True)
    self.log('val acc', self.accuracy(output, label.int()))
    return loss

  def test_step(self, batch, batch_idx):
    input_ids =batch['input_ids']
    attention_mask = batch['attention_mask']
    label = batch['label']
    loss , output = self(input_ids, attention_mask, label)
    self.log('test loss', loss, prog_bar=True, logger=True)
    self.log('test acc', self.accuracy(output, label.int()))
    return loss

  def training_epoch_end(self, output):
    self.log('train_acc_epoch', self.accuracy.compute())

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=2e-5)
    warmup_steps = self.steps_per_epochs//3
    total_steps = self.steps_per_epochs*self.n_epochs-warmup_steps

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps
    )  
    return [optimizer],[scheduler]