import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from torchvision import transforms
from PIL import Image

torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image_path',
                    type=str,
                    required=True)

parser.add_argument('-s', '--state_dict_path',
                    type=str,
                    required=True)

parser.add_argument('-d', '--dataset',
                    help="Options: 'afad', 'morph2', or 'cacd'.",
                    type=str,
                    required=True)

args = parser.parse_args()
IMAGE_PATH = args.image_path
STATE_DICT_PATH = args.state_dict_path
GRAYSCALE = False

if args.dataset == 'afad':
    NUM_CLASSES = 26
    ADD_CLASS = 15

elif args.dataset == 'morph2':
    NUM_CLASSES = 55
    ADD_CLASS = 16

elif args.dataset == 'cacd':
    NUM_CLASSES = 49
    ADD_CLASS = 14

else:
    raise ValueError("args.dataset must be 'afad',"
                     " 'morph2', or 'cacd'. Got %s " % (args.dataset))


############################
### Load image
############################


image = Image.open(IMAGE_PATH)
custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.CenterCrop((120, 120)),
                                       transforms.ToTensor()])
image = custom_transform(image)
DEVICE = torch.device('cpu')
image = image.to(DEVICE)


##########################
# MODEL
##########################


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.num_classes = num_classes
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, (self.num_classes-1)*2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        logits = logits.view(-1, (self.num_classes-1), 2)
        probas = F.softmax(logits, dim=2)[:, :, 1]
        return logits, probas


def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model


#######################
### Initialize Model
#######################

model = resnet34(NUM_CLASSES, GRAYSCALE)
model.load_state_dict(torch.load(STATE_DICT_PATH, map_location=DEVICE))
model.eval()

image = image.unsqueeze(0)

with torch.set_grad_enabled(False):
    logits, probas = model(image)
    predict_levels = probas > 0.5
    predicted_label = torch.sum(predict_levels, dim=1)
    print('Class probabilities:', probas)
    print('Predicted class label:', predicted_label.item())
    print('Predicted age in years:', predicted_label.item() + ADD_CLASS)
