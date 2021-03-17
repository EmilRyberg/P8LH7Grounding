import torch
from torch import nn
from collections import OrderedDict

# Load pretrained model
model = torch.hub.load('pytorch/vision:v0.7.0', 'mobilenet_v2', pretrained=True)

# Remove classification FC layer
classifier_name, old_classifier = model._modules.popitem()

# Freeze layers for transfer learning
for param in model.parameters():
    param.requires_grad = False


# Add new classification (feature) layer
classifier_input_size = old_classifier[1].in_features
output_layer_size = 128

classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(classifier_input_size, output_layer_size)),
                           ('output', nn.Softmax(dim=1))
                           ]))

model.add_module(classifier_name, classifier)


print(model.classifier.parameters())

torch.save(model, "MNV2_1.pt")

model2 = torch.load("MNV2_1.pt")