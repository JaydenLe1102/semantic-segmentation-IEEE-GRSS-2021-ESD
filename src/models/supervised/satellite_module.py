import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torchmetrics

from src.models.supervised.segmentation_cnn import SegmentationCNN
from src.models.supervised.unet import UNet
from src.models.supervised.resnet_transfer import FCNResnetTransfer
from src.models.supervised.deeplabv3 import DeepLabV3Plus
from src.models.supervised.deeplabv3_unet import DeepLabV3_Unet
from src.models.supervised.attention_unet import AttentionUNet
from src.models.supervised.deeplabv3_attention_unet import DeepLabV3_AttentionUnet

class ESDSegmentation(pl.LightningModule):
    """
    LightningModule for training a segmentation model on the ESD dataset
    """
    def __init__(self, model_type, in_channels, out_channels, 
                 learning_rate=1e-3, model_params: dict = {}):
        """
        Initializes the model with the given parameters.

        Input:
        model_type (str): type of model to use, one of "SegmentationCNN",
        "UNet", or "FCNResnetTransfer"
        in_channels (int): number of input channels of the image of shape
        (batch, in_channels, width, height)
        out_channels (int): number of output channels of prediction, prediction
        is shape (batch, out_channels, width//scale_factor, height//scale_factor)
        learning_rate (float): learning rate of the optimizer
        model_params (dict): dictionary of parameters to pass to the model

        """
        super().__init__()
        self.save_hyperparameters()
        
        # define performance metrics for segmentation task
        # such as accuracy per class accuracy, average IoU, per class IoU,
        # per class AUC, average AUC, per class F1 score, average F1 score
        # these metrics will be logged to weights and biases
        
        if model_type == "SegmentationCNN":
            self.model = SegmentationCNN(in_channels, out_channels, **model_params)
        elif model_type == "UNet":
            self.model = UNet(in_channels, out_channels, **model_params)
        elif model_type == "FCNResnetTransfer":
            self.model = FCNResnetTransfer(in_channels, out_channels, **model_params)
        elif model_type == "DeepLabV3_Unet":
            self.model = DeepLabV3_Unet(in_channels, out_channels, **model_params)
        elif model_type == "DeepLabV3Plus":
            self.model = DeepLabV3Plus(in_channels, out_channels, **model_params)
        elif model_type == "AttentionUNet":
            self.model = AttentionUNet(in_channels, out_channels, **model_params)
        elif model_type == "DeepLabV3_AttentionUnet":
            self.model = DeepLabV3_AttentionUnet(in_channels, out_channels, **model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=out_channels)
        self.loss_fn = nn.CrossEntropyLoss()

        self.val_iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=out_channels)
        self.val_roc_auc = torchmetrics.AUROC(task="multiclass", num_classes=out_channels)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=out_channels, average='none')
        self.val_f1_average = torchmetrics.F1Score(task="multiclass", num_classes=out_channels, average='macro')
    
    def forward(self, X):
        """
        Run the input X through the model

        Input: X, a (batch, input_channels, width, height) image
        Ouputs: y, a (batch, output_channels, width/scale_factor, height/scale_factor) image
        """
        return self.model(X)
    
    def training_step(self, batch, batch_idx):
        """
        Gets the current batch, which is a tuple of
        (sat_img, mask, metadata), predicts the value with
        self.forward, then uses CrossEntropyLoss to calculate
        the current loss.

        Note: CrossEntropyLoss requires mask to be of type
        torch.int64 and shape (batches, width, height), 
        it only has one channel as the label is encoded as
        an integer index. As these may not be this shape and
        type from the dataset, you might have to use
        torch.reshape or torch.squeeze in order to remove the
        extraneous dimensions, as well as using Tensor.to to
        cast the tensor to the correct type.

        Note: The type of the tensor input to the neural network
        must be the same as the weights of the neural network.
        Most often than not, the default is torch.float32, so
        if you haven't casted the data to be float32 in the
        dataset, do so before calling forward.

        Input:
            batch: tuple containing (sat_img, mask, metadata).
                sat_img: Batch of satellite images from the dataloader,
                of shape (batch, input_channels, width, height)
                mask: Batch of target labels from the dataloader,
                by default of shape (batch, 1, width, height)
                metadata: List[SubtileMetadata] of length batch containing 
                the metadata of each subtile in the batch. You may not
                need this.

            batch_idx: int indexing the current batch's index. You may
            not need this input, but it's part of the class' interface.

        Output:
            train_loss: torch.tensor of shape (,) (i.e, a scalar tensor).
            Gradients will not propagate unless the tensor is a scalar tensor.
        """
        sat_img, mask, _ = batch
        sat_img = sat_img.float()
        mask = mask.squeeze().long()
        preds = self.forward(sat_img)
        loss = self.loss_fn(preds, mask)
        acc = self.train_accuracy(preds, mask)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'train_acc': acc}
    
    
    def validation_step(self, batch, batch_idx):
        """
        Gets the current batch, which is a tuple of
        (sat_img, mask, metadata), predicts the value with
        self.forward, then evaluates the 

        Note: The type of the tensor input to the neural network
        must be the same as the weights of the neural network.
        Most often than not, the default is torch.float32, so
        if you haven't casted the data to be float32 in the
        dataset, do so before calling forward.

        Input:
            batch: tuple containing (sat_img, mask, metadata).
                sat_img: Batch of satellite images from the dataloader,
                of shape (batch, input_channels, width, height)
                mask: Batch of target labels from the dataloader,
                by default of shape (batch, 1, width, height)
                metadata: List[SubtileMetadata] of length batch containing 
                the metadata of each subtile in the batch. You may not
                need this.

            batch_idx: int indexing the current batch's index. You may
            not need this input, but it's part of the class' interface.

        Output:
            val_loss: torch.tensor of shape (,) (i.e, a scalar tensor).
            Should be the cross_entropy_loss, as it is the main validation
            loss that will be tracked.
            Gradients will not propagate unless the tensor is a scalar tensor.
        """

        sat_img, mask, _ = batch 
        sat_img = sat_img.float()
        mask = mask.squeeze().long()
        preds = self.forward(sat_img)
        val_loss = self.loss_fn(preds, mask)
        acc = self.val_accuracy(preds, mask)
        iou = self.val_iou(preds, mask)
        f1 = self.val_f1(preds, mask)
        f1_avg = self.val_f1_average(preds, mask)

        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_iou', iou, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', f1.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1_avg', f1_avg, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        preds_softmax = torch.softmax(preds, dim=1)
        val_roc_auc = self.val_roc_auc(preds_softmax, mask)
        self.log('val_roc_auc', val_roc_auc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'val_loss': val_loss, 'val_acc': acc, 'val_iou': iou, 'val_f1': f1, 'val_f1_avg': f1_avg, 'val_roc_auc': val_roc_auc}
        
    
    def configure_optimizers(self):
        """
        Loads and configures the optimizer. See torch.optim.Adam
        for a default option.

        Outputs:
            optimizer: torch.optim.Optimizer
                Optimizer used to minimize the loss
        """
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)

        return optimizer
