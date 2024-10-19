from typing import Any

from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from transformers import SwinModel, AutoImageProcessor
import lightning as L
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import torch


class Swin4MRI(L.LightningModule):

    def __init__(self, num_classes, use_cls_token: bool, freeze_percentage=70):
        super().__init__()
        self.encoder = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
        self.use_cls_token = use_cls_token
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.l1 = torch.nn.Linear(49, num_classes)
        self.l2 = torch.nn.Linear(1024, num_classes)
        self.freeze_model_layers(self, freeze_percentage)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    def compute_loss(self, predicted, labels):
        return self.criterion(predicted, labels)

    def forward(self, x):
        x = self.encoder(x).last_hidden_state
        if self.use_cls_token:
            # use cls_token -> result : torch.size([batch_size,49])
            x = x[:, 0, :]
            x = self.l2(x)
        else:
            # use average pool -> result : torch.size([batch_size,1024])
            x = self.avg_pool(x).squeeze(2)
            x = self.l1(x)
        return x

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        X, y = batch
        predicted_logits = self.forward(X)
        loss = self.compute_loss(predicted_logits, y)

        self.log('train_loss', loss,on_step=False, on_epoch=True,logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch

        predicted_logits = self.forward(X)

        # Get the predicted class indices by applying softmax and argmax
        predicted_classes = torch.argmax(predicted_logits, dim=1)
        # Calculate loss for evaluation
        loss = self.criterion(predicted_logits, y).item()  # Compute loss

        # Compute metrics
        f1 = f1_score(y.cpu().numpy(), predicted_classes.detach().cpu().numpy(), average='macro')
        acc = accuracy_score(y.cpu().numpy(), predicted_classes.detach().cpu().numpy())
        self.log_dict({'eval_loss': loss, "f1_score": f1, "acc": acc}, prog_bar=True)  # Log evaluation loss

        self.log('test_loss', loss)

    def test_step(self, batch, batch_idx):
        X, y = batch

        predicted_logits = self.forward(X)

        # Get the predicted class indices by applying softmax and argmax
        predicted_classes = torch.argmax(predicted_logits, dim=1)
        # Calculate loss for evaluation
        loss = self.criterion(predicted_logits, y).item()  # Compute loss

        # Compute metrics
        f1 = f1_score(y.cpu().numpy(), predicted_classes.detach().cpu().numpy(), average='macro')
        acc = accuracy_score(y.cpu().numpy(), predicted_classes.detach().cpu().numpy())
        self.log('train_loss', loss)

    @staticmethod
    def freeze_model_layers(model, freeze_percentage=70):
        """
        Freeze a given percentage of layers in a Hugging Face model.
        Args:
            model: Hugging Face model (e.g., BertModel, GPT2Model, etc.)
            freeze_percentage: Percentage of layers to freeze (0 to 100).
        """

        def count_parameters(model):
            """Calculate the total number of parameters and trainable parameters."""
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"Total Parameters: {total_params:,}")
            print(f"Trainable Parameters: {trainable_params:,}")

            return total_params, trainable_params

        # Get the total number of parameters (or layers)
        total_layers = sum(1 for _ in model.parameters())

        # Calculate how many parameters (or layers) to freeze
        num_to_freeze = int((freeze_percentage / 100) * total_layers)

        # Iterate through the model parameters
        for i, param in enumerate(model.parameters()):
            if i < num_to_freeze:
                param.requires_grad = False  # Freeze this parameter
            else:
                param.requires_grad = True  # Keep it trainable

        total_params, trainable_params = count_parameters(model)
        print(f"Frozen {num_to_freeze}/{total_layers} layers.")
        print(f"trainable parameters number = {trainable_params}/{total_params}")
