import torch.nn as nn
import torchvision.models as models


class LocalOccNet(nn.Module):
    """ Implement the local occupancy predictor
        Input: a panoramic image observation
        Output: local occupancy
    """
    def __init__(self, configs):
        super(LocalOccNet, self).__init__()
        # Network configurations
        self.configs = configs if configs is not None else {"dropout" : 0.5, "use_pretrained_resnet18" : True}

        # Define the visual encoder: use convolutional layers in Resnet18
        self.conv_layer = nn.Sequential(*self.obtain_resnet18_conv_layers())

        # Define the latent feature encoder: convolutional layer
        self.conv_br_layer = nn.Sequential(
            nn.Conv2d(512, 512, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, (1, 1), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Define the latent feature encoder: Linear layer
        self.fc_proj_layer = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(self.configs['dropout']),

            nn.Linear(1024, 4096),
            nn.ReLU(inplace=True)
        )

        # Define the deconvolutional layer
        self.de_conv_layer = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 32, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, (1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )

    def obtain_resnet18_conv_layers(self):
        layers = []
        if self.configs['use_pretrained_resnet18']:
            try:
                # For PyTorch version > 0.2.0
                resnet18 = models.resnet18(weights='ResNet18_Weights.DEFAULT')
            except TypeError:
                # For PyTorch version earlier
                resnet18 = models.resnet18(pretrained=True)
        else:
            resnet18 = models.resnet18()
        for name, param in resnet18.named_children():
            # Break when fully connected layer is met
            if name == "fc":
                break
            # Only save the convolutional layers
            layers.append(param)
        return layers

    def forward(self, x):
        # Compute the size of the input data
        bs, c, h, w = x.size()

        # Compute the convolutional feature
        x = self.conv_layer(x)

        # Convolutional layer
        x = self.conv_br_layer(x).view(bs, -1)

        # Perform the projection
        x = self.fc_proj_layer(x).view(bs, 64, 8, 8)

        # Reconstruct the local occupancy
        x = self.de_conv_layer(x)

        return x
