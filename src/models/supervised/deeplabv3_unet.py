import torch
import torch.nn as nn
from src.models.supervised.deeplabv3 import DeepLabV3Plus
from src.models.supervised.unet import UNet


class DeepLabV3_Unet(nn.Module):
  
  def __init__ (self, in_channels, out_channels, scale_factor=50, **kwargs):
    
    super(DeepLabV3_Unet, self).__init__()
    
    self.in_channels = in_channels
    self.out_channels = out_channels
    
    self.deeplabv3 = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
    self.deeplabv3.backbone.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.deeplabv3.classifier[-1] = nn.Conv2d(256, self.in_channels, kernel_size=1, stride=1)

    self.unet = UNet(in_channels, out_channels, **kwargs)
    
  
  def forwardDeepLabV3(self, x):
				
			x = self.deeplabv3(x)['out']
   
			return x
    

  def forward(self, x):
    
    x = self.forwardDeepLabV3(x)
    x = self.unet(x)
		
    return x