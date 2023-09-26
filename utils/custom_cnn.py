
from torch import nn
import warnings
warnings.filterwarnings("ignore")



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        #x = self.conv(x)
        x = self.pool( self.relu( self.conv(x)  ) )
        return x
    
    
class SingleCNN2D(nn.Module):
    """
    2D CNN which takes Mel Spectrograms as inputs.
    No hyperparameters allowed.
    
    Only for Imagimob case study
    """
    def __init__(self):
        super().__init__()
        
        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(19200, 1)
        # self.sgm = nn.Sigmoid() Not needed if using nn.BCELogitsLoss() because it is embedded there
        
    def forward(self, x):
        x = self.flatten( self.conv4( self.conv3( self.conv2( self.conv1( x ) ) ) ) )
        # x = self.linear(x)
        return self.linear(x)