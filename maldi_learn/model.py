# Import modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


# implement pytorch lightning model 
class MALDI_1D_CNN(pl.LightningModule):
    """
    MALDI_1D_CNN is a PyTorch Lightning module for training a 1D convolutional neural network (CNN)
    on MALDI-TOF mass spectrometry data.

    Args:
        classes (int): The number of classes in the classification task.
        backbone (nn.Module): The backbone CNN model.
        params (dict): A dictionary of hyperparameters for the model.
        cosine_t_max (float): The maximum value of the cosine margin for the angular softmax loss.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.0003.

    Attributes:
        backbone (nn.Module): The backbone CNN model.
        params (dict): A dictionary of hyperparameters for the model.
        learning_rate (float): The learning rate for the optimizer.
        cosine_t_max (float): The maximum value of the cosine margin for the angular softmax loss.
        loss (function): The loss function based on the task type.
        train_acc (torchmetrics.Accuracy): The accuracy metric for training.
        val_acc (torchmetrics.Accuracy): The accuracy metric for validation.
        test_acc (torchmetrics.Accuracy): The accuracy metric for testing.
    """

    def __init__(self, classes, backbone, params, cosine_t_max, learning_rate=0.0003):
        super().__init__()
        self.backbone = backbone(classes,
                                 cnn1_kernel=params["cnn1_kernel"],
                                 fc1_nodes=params["fc1_nodes"],
                                 max_pool_kernel=params["max_pool_kernel"],
                                 max_pool_stride=params["max_pool_stride"],
                                 dropout=params["dropout"])
        self.params = params
        self.learning_rate = learning_rate
        self.cosine_t_max = cosine_t_max

        if self.params["task"] == "multiclass":
            self.loss = F.cross_entropy
        if self.params["task"] == "multilabel":
            self.loss = F.binary_cross_entropy_with_logits
        # class_weights = torch.tensor(self.params["class_weights"]).to(self.device)
        # self.loss = lambda input, target: F.cross_entropy(input, target, weight=class_weights)
        elif self.params["task"] == "binary":
            self.loss = F.binary_cross_entropy_with_logits

        self.save_hyperparameters(ignore=['backbone'])

        # Select metric from pytorch metrics
        self.train_acc = torchmetrics.Accuracy(task=self.params["task"], num_classes=classes, num_labels=classes)
        self.val_acc   = torchmetrics.Accuracy(task=self.params["task"], num_classes=classes, num_labels=classes)
        self.test_acc  = torchmetrics.Accuracy(task=self.params["task"], num_classes=classes, num_labels=classes)

    def forward(self, x):
        return self.backbone(x)
    
    def _shared_step(self, batch):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.loss(y_hat, y, weight = self.params["class_weights"])
        if self.params["task"] in "multiclass or binary":
            preds = torch.argmax(y_hat, dim=1)
            labels = y.argmax(dim=1)
        else:
            preds = (y_hat>0.5).int()
            labels = y
        return loss, preds, labels
    
    def training_step(self, batch, batch_idx):
        
        loss, preds, labels = self._shared_step(batch)
        self.train_acc(preds, labels)
        self.log('train_loss', loss)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.val_acc(preds, labels)
        self.log('val_loss', loss)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.test_acc(preds, labels)
        self.log('test_loss', loss)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # implement learning rate scheduler
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cosine_t_max)
        return {
            "optimizer": optimizer
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "interval": "step"
            # }
        }
    



# 1D CNN model for the MALDI data
class backboneModel(nn.Module):
    """
    Backbone model for classification tasks using 1D Convolutional Neural Networks.

    Args:
        num_classes (int): Number of output classes.
        cnn1_kernel (int): Kernel size for the first convolutional layer.
        fc1_nodes (int): Number of nodes in the first fully connected layer.
        max_pool_kernel (int): Kernel size for the max pooling layer.
        max_pool_stride (int): Stride for the max pooling layer.
        dropout (float): Dropout rate.

    Attributes:
        conv1 (nn.Conv1d): First convolutional layer.
        bn1 (nn.BatchNorm1d): Batch normalization layer after the first convolutional layer.
        conv2 (nn.Conv1d): Second convolutional layer.
        bn11 (nn.BatchNorm1d): Batch normalization layer after the second convolutional layer.
        fc1 (nn.Linear): First fully connected layer.
        bn2 (nn.BatchNorm1d): Batch normalization layer after the first fully connected layer.
        fc2 (nn.Linear): Output layer.
        dropout (nn.Dropout): Dropout layer.
        pool (nn.MaxPool1d): Max pooling layer.
        activation (nn.SiLU): Activation function.

    """

    def __init__(self, num_classes, cnn1_kernel, fc1_nodes, max_pool_kernel, max_pool_stride, dropout):
        super().__init__()

        self.cnn1_kernel = cnn1_kernel
        self.fc1_nodes = fc1_nodes
        self.max_pool_kernel = max_pool_kernel
        self.max_pool_stride = max_pool_stride
        self.dropout = dropout

        self.conv1 = nn.Conv1d(1, 8, kernel_size=self.cnn1_kernel, stride=1)
        self.bn1 = nn.BatchNorm1d(8)
        
        self.conv2 = nn.Conv1d(8, 4, self.max_pool_kernel, stride=self.max_pool_stride)
        self.bn11 = nn.BatchNorm1d(4)

        
        self.fc1 = nn.Linear(self.fc1_nodes, 128)#
        self.bn2 = nn.BatchNorm1d(1)
        self.fc2 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(self.dropout)
        self.pool = nn.MaxPool1d(self.max_pool_kernel, stride=self.max_pool_stride)
        self.activation = nn.SiLU()
    
    def forward(self, x):

        # 1D Conv layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        # Pooling 1D Conv layer
        x = self.conv2(x)
        x = self.bn11(x)
        x = self.activation(x)
        x = x.reshape(x.shape[0], 1, -1)

        # Dynamic check for fc1 input features
        if x.shape[-1] != self.fc1.in_features:
            device = next(self.fc1.parameters()).device
            self.fc1 = nn.Linear(x.shape[-1], 128).to(device)

        # Fully connected layer
        x = self.fc1(x)
        x = self.bn2(x)
        x = self.activation(x)

        # Output layer
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.squeeze(1)
