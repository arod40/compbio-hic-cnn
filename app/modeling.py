import torch.nn as nn
import torchvision
import torch

class SiameseNetwork(nn.Module):
    """
        Based on https://github.com/pytorch/examples/tree/main/siamese_network
        Siamese network for image similarity estimation.
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
        `"FaceNet" <https://arxiv.org/pdf/1503.03832.pdf>`_ is a variant of the Siamese network.
        This implementation varies from FaceNet as we use the `ResNet-18` model from
        `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_ as our feature extractor.
        In addition, we aren't using `TripletLoss`, `BCELoss` can do the trick.
    """
    def __init__(self, input_size):
        super(SiameseNetwork, self).__init__()

        self.encoder, encoded_features = self.build_encoder(input_size)

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(encoded_features * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.encoder.apply(self.init_weights)
        self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def build_encoder(self, input_size, **kwargs):
       raise NotImplementedError()


    def forward_once(self, x):
        output = self.encoder(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        # concatenate both images' features
        output = torch.cat((output1, output2), 1)

        # pass the concatenation to the linear layers
        output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        output = self.sigmoid(output)
        
        return output

class SiameseNetworkResnetEncoder(SiameseNetwork):
    def __init__(self, input_size):
        super(SiameseNetworkResnetEncoder, self).__init__(input_size)


    def build_encoder(self, input_size):
        # get resnet model
        resnet = torchvision.models.resnet18(pretrained=False)

        # over-write the first conv layer to be able to read HIC images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas HIC has (1,x,x) where 1 is a gray-scale channel
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))

        return resnet, resnet(torch.rand(1, 1, *input_size)).shape[1]

class SiameseNetworkLinearEncoder(SiameseNetwork):
    def __init__(self, input_size):
        super(SiameseNetworkLinearEncoder, self).__init__(input_size)

    def build_encoder(self, input_size):
        return nn.Linear(input_size[0] * input_size[1], 256), 256
     
    def forward_once(self, x):
        output = self.encoder(x.view(x.size()[0], -1))
        return output
    
class SiameseNetworkLeNetEncoder(SiameseNetwork):
    def __init__(self, input_size):
        super(SiameseNetworkLeNetEncoder, self).__init__(input_size)

    def build_encoder(self, input_size):
        encoder = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )

        return encoder, torch.prod(torch.tensor(encoder(torch.rand(1, 1, *input_size)).shape))