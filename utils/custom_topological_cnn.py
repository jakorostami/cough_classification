
from torch import nn
import warnings
warnings.filterwarnings("ignore")



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2, mc_dropout=True, mc_rate=0.2):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        if mc_dropout:
            self.mcdrop = True
            self.dropout = nn.Dropout2d(mc_rate)
            
        
    def forward(self, x):
        #x = self.conv(x)
        x = self.pool( self.relu( self.conv(x)  ) )
        if self.mcdrop:
            return self.dropout(x)
        else:
            return x
    
    
class SingleCNN2DTopo(nn.Module):
    """
    2D CNN which takes topological embeddings as inputs.
    No hyperparameters allowed.
    
    Only for Imagimob case study
    """
    def __init__(self, mc_dropout=True, mc_rate=.2):
        super().__init__()
        
        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)
        
        self.flatten = nn.Flatten()
        if mc_dropout:
            self.mcdrop = True
            self.dropout = nn.Dropout(mc_rate)
        
        self.linear = nn.Linear(481280, 1)
        # self.sgm = nn.Sigmoid() Not needed if using nn.BCELogitsLoss() because it is embedded there
        
    def forward(self, x):
        x = self.flatten( self.conv4( self.conv3( self.conv2( self.conv1( x ) ) ) ) )
        if self.mcdrop:
            x = self.dropout(x)
            return self.linear(x)
        # x = self.linear(x)
        ### DEBUGGING MODE BELOW
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x = self.flatten(x)
        # x = self.linear(x)
        else:
            return self.linear(x)