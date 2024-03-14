
import torch
import torch.nn as nn


class DeepLabV3Plus(nn.Module):
		def __init__(self, in_channels, out_channels, scale_factor=50, **kwargs):
				
			super(DeepLabV3Plus, self).__init__()
   
			self.model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
			self.input_channels = int(in_channels)
			self.output_channels = int(out_channels)
    
			self.model.backbone.conv1 = nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

			self.model.classifier[-1] = nn.Conv2d(256, self.output_channels, kernel_size=1, stride=1)
			
			self.pool = nn.AvgPool2d(scale_factor)


		def forward(self, x):
				
			x = self.model(x)['out']
			y = self.pool(x)
			return y
 
 
# 78.6133% accuracy with epoch 27 and default parameters (metrics7)