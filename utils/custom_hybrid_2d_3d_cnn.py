

from torch import nn
import torch
import warnings
warnings.filterwarnings("ignore")


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2):
        super(ConvBlock3D, self).__init__()
        
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=2)
        
    def forward(self, x):
        #x = self.conv(x)
        x = self.pool( self.relu( self.conv(x)  ) )
        return x
    
class ConvBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2):
        super(ConvBlock2D, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        
    def forward(self, x):
        #x = self.conv(x)
        x = self.pool( self.relu( self.conv(x)  ) )
        return x
    
    
class Conv3DModel(nn.Module):
    def __init__(self):
        super(Conv3DModel, self).__init__()
        
        self.conv3d_1 = ConvBlock3D(1, 128)
        self.conv3d_2 = ConvBlock3D(128, 256)
        self.drop3d_1 = nn.Dropout3d(.5)
        self.batchnorm3d = nn.BatchNorm3d(256)
        self.conv3d_3 = ConvBlock3D(256, 256)
        self.conv3d_4 = ConvBlock3D(256, 512)
        self.drop3d_2 = nn.Dropout3d(.5)
        self.batchnorm3d_2 = nn.BatchNorm3d(512)
        self.conv3d_5 = ConvBlock3D(512, 512)
        self.flat3d = nn.Flatten()
        
    
    def forward(self, x):
        x = self.batchnorm3d( self.drop3d_1( self.conv3d_2( self.conv3d_1( x ) ) ) )
        x = self.batchnorm3d_2( self.drop3d_2( self.conv3d_4( self.conv3d_3( x  ) ) ) )
        x = self.flat3d( self.conv3d_5( x )) 
        return x


class Conv2DModel(nn.Module):
    def __init__(self):
        super(Conv2DModel, self).__init__()
 
        self.conv2d_1 = ConvBlock2D(1, 32)
        self.conv2d_2 = ConvBlock2D(32, 64)
        self.drop2d_1 = nn.Dropout2d(0.5)
        self.batchnorm2d = nn.BatchNorm2d(64)
        self.conv2d_3 = ConvBlock2D(64, 256)
        self.conv2d_4 = ConvBlock2D(256, 512)
        self.drop2d_2 = nn.Dropout2d(0.5)
        self.batchnorm2d_2 = nn.BatchNorm2d(512)
        self.conv2d_5 = ConvBlock2D(512, 512)
        self.flat2d = nn.Flatten()       
        
        
    def forward(self, x):
        x = self.batchnorm2d( self.drop2d_1( self.conv2d_2( self.conv2d_1( x ) ) ) )
        x = self.batchnorm2d_2( self.drop2d_2( self.conv2d_4( self.conv2d_3( x )  ) ) )
        x = self.flat2d( self.conv2d_5( x ) ) 
        return x
        


class HybridCNN(nn.Module):
    def __init__(self):
        super(HybridCNN, self).__init__()
        
        # Create 3D CNN part
        self.block3d = Conv3DModel()
        
        # Create 2D CNN part
        self.block2d = Conv2DModel()
        
        # Use linear layers (fully connected)
        self.linear1 = nn.Linear(in_features=(16384), out_features=32)  
        self.linear2 = nn.Linear(32, 1)
        
    def forward(self, x3d, x2d):

        x3d = self.block3d(x3d)
        x2d = self.block2d(x2d)
        
        # Combine their output
        x = torch.cat((x3d, x2d), dim=1)
        
        # Send them through linear layers
        x = self.linear1(x)
        x = self.linear2(x)
        
        return x