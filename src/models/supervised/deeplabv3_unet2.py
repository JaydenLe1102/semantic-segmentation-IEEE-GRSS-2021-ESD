import torch
import torch.nn as nn
from src.models.supervised.deeplabv3 import DeepLabV3Plus
from src.models.supervised.unet import Encoder, Decoder, DoubleConvHelper

class DeepLabV3_Unet(nn.Module):
  
  def __init__ (self, in_channels, out_channels, n_encoders: int = 2, embedding_size: int = 64, scale_factor=50,  **kwargs):
    
    super(DeepLabV3_Unet, self).__init__()
    in_channels = int(in_channels)
    out_channels = int(out_channels)        

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.n_encoders = n_encoders
    self.embedding_size = embedding_size
    self.scale_factor = scale_factor
    
    self.deeplabv3 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    
    deepLabOut_channels = embedding_size * 2 ** (n_encoders)
    self.scale_facDeepLab = (n_encoders) * 2
    
    self.deeplabpool = nn.AvgPool2d(self.scale_facDeepLab)
    
    
    self.deeplabv3.backbone.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.deeplabv3.classifier[-1] = nn.Conv2d(256, deepLabOut_channels, kernel_size=1, stride=1)
    
    self.first_layer = DoubleConvHelper(in_channels, embedding_size)

    # Encoder blocks
    self.encoders = nn.ModuleList()
    for i in range(n_encoders - 1):  # Use n_encoders for correct iteration
        in_enc_channels =  self.embedding_size * 2**i
        out_enc_channels = self.embedding_size * 2**(i + 1)
        self.encoders.append(
            Encoder(in_enc_channels, out_enc_channels)
        )
        
        
    # Decoder blocks
    self.decoders = nn.ModuleList()
    for i in range(n_encoders):  # Use n_encoders - 1 for correct decoder count
        in_dec_channels = self.embedding_size * 2**(n_encoders - i)
        out_dec_channels = self.embedding_size * 2**(n_encoders - i - 1)
        self.decoders.append(
            Decoder(in_dec_channels, out_dec_channels)
        )

    # Final convolution layer
    self.final_conv = nn.Conv2d(
        self.embedding_size, self.out_channels, kernel_size=1
    )

    # Pooling layer for output downscaling
    self.pool = nn.MaxPool2d(scale_factor, stride=scale_factor)
    
  
  def forwardDeepLabV3(self, x):
    x = self.deeplabv3(x)['out']
    x = self.deeplabpool(x)
    return x
    

  def forward(self, x):
    
    x1 = self.forwardDeepLabV3(x)
    
    
    x = self.first_layer(x)
      

    # Encoder pass
    encoder_outputs = []
    encoder_outputs.append(x)
    for i in range(len(self.encoders)):
        x = self.encoders[i](x)
        encoder_outputs.append(x)
       
    x = x1

    # Decoder pass
    for i in range(len(self.decoders)):
        encoder_out = encoder_outputs[-(i + 1)]
        x = self.decoders[i](x, encoder_out)

    # Final convolution
    x = self.final_conv(x)

    # Downscale output
    x = self.pool(x)

    return x