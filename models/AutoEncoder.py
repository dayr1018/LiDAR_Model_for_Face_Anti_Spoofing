import torch
import torch.nn as nn
from skimage.util import random_noise

# There are 3 types of Autoencoder
# 1. Model using RGB 
# 2. Model using Depth 
# 3. Model using RGB+Depth

class AutoEncoder_RGB(nn.Module):
    def __init__(self, layer_number=3, batch_norm=False, dropout_rate=0):
        super(AutoEncoder_RGB, self).__init__()

        # dropout parameter
        self.dropout_rate = dropout_rate
        if self.dropout_rate is not 0:
            self.dropout_layer = nn.Dropout2d(dropout_rate)

        self.encoder = make_encoder(layer_number, 3, batch_norm, nn.ELU())
        self.decoder = make_decoder(layer_number, 3, batch_norm, nn.ELU())
    
    def forward(self, x):

        if self.dropout_rate is not 0:
            x = self.dropout_layer(x)

        latent = self.encoder(x)
        out = self.decoder(latent)
      
        return out

class AutoEncoder_Depth(nn.Module):
    def __init__(self, layer_number=3, batch_norm=False, dropout_rate=0):
        super(AutoEncoder_Depth, self).__init__()

        # dropout parameter
        self.dropout_rate = dropout_rate
        if self.dropout_rate is not 0:
            self.dropout_layer = nn.Dropout2d(dropout_rate)

        self.encoder = make_encoder(layer_number, 1, batch_norm, nn.ELU())
        self.decoder = make_decoder(layer_number, 1, batch_norm, nn.ELU())
    
    def forward(self, x):

        if self.dropout_rate is not 0:
            x = self.dropout_layer(x)

        latent = self.encoder(x)
        out = self.decoder(latent)
      
        return out

class AutoEncoder_Intergrated_Basic(nn.Module):
    def __init__(self, layer_number=4, batch_norm=False, dropout_rate=0):
        super(AutoEncoder_Intergrated_Basic, self).__init__()

        # dropout parameter
        self.dropout_rate = dropout_rate
        if self.dropout_rate is not 0:
            self.dropout_layer = nn.Dropout2d(dropout_rate)

        self.encoder = make_encoder(layer_number, 4, batch_norm, nn.ELU())
        self.decoder = make_decoder(layer_number, 4, batch_norm, nn.ELU())
    
    def forward(self, x, y):

        input_tensor = torch.cat((x, y), dim=1)  

        if self.dropout_rate is not 0:
            input_tensor = self.dropout_layer(input_tensor)

        latent = self.encoder(input_tensor)
        out = self.decoder(latent)
      
        return out

class AutoEncoder_Intergrated_Proposed(nn.Module):
    def __init__(self, rlayer_number=3, dlayer_number=5, batch_norm=False, dropout_rate=0):
        super(AutoEncoder_Intergrated_Proposed, self).__init__()

        # dropout parameter
        self.dropout_rate = dropout_rate
        if self.dropout_rate is not 0:
            self.dropout_layer = nn.Dropout2d(dropout_rate)
 
        # RGB Autoencoder: layer 3, Using Gaussiann Noise(0.01), Not using dropout layer
        self.rgb_encoder = make_encoder(rlayer_number, 3, batch_norm, nn.ELU())
        self.rgb_decoder = make_decoder(rlayer_number, 3, batch_norm, nn.ELU())
        
        # Depth Autoencoder : layer 5, Using Dropout layer(0.7), Not using Gaussian Noise
        self.depth_encoder = make_encoder(dlayer_number, 1, batch_norm, nn.ELU())
        self.depth_decoder = make_decoder(dlayer_number, 1, batch_norm, nn.ELU())
       
    def forward(self, x, y):

        # RGB Autoencoder
        latent1 = self.rgb_encoder(x)
        out1 = self.rgb_decoder(latent1)
      
        # Depth Autoencoder
        if self.dropout_rate is not 0:
            y = self.dropout_layer(y)
        latent2 = self.depth_encoder(y)
        out2 = self.depth_decoder(latent2)
      
        out = torch.cat((out1, out2), dim=1)
        return out

def make_encoder(layer_number, input_channel, batch_norm, act_func):
    encoder = []

    # 첫번째 층
    encoder.append(make_encoder_layer(input_channel, 16, batch_norm, act_func))
    # 2~마지막 층
    for i in range(1, layer_number):
        encoder.append(make_encoder_layer(16*i, 16*(i+1), batch_norm, act_func))

    return nn.Sequential(*encoder)
    
def make_decoder(layer_number, output_channel, batch_norm, act_func):
    decoder = []

    # 1~(마지막-1) 층
    for i in range(layer_number, 1, -1):
        decoder.append(make_decoder_layer(16*i, 16*(i-1), batch_norm, act_func))
    # 마지막 층
    decoder.append(make_decoder_layer(16, output_channel, False, nn.Sigmoid()))

    return nn.Sequential(*decoder)


def make_encoder_layer(in_channel, out_channel, batch_norm, act_func):
    layers = []

    layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1))
    if batch_norm: 
        layers.append(nn.BatchNorm2d(out_channel))
    layers.append(act_func)
    layers.append(nn.MaxPool2d(2,2))

    return nn.Sequential(*layers)

def make_decoder_layer(in_channel, out_channel, batch_norm, act_func):
    layers = []

    layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2))
    if batch_norm :
        layers.append(nn.BatchNorm2d(out_channel))
    layers.append(act_func)

    return nn.Sequential(*layers)
