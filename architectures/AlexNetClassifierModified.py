import torch
import torch.nn as nn

#my modified architecture with added relu & linear layer
class AlexNetClassifier(nn.Module):
    def __init__(self, num_classes=4, num_feat=2048):
        super(AlexNetClassifier, self).__init__()
        self.fc_classifier = nn.Sequential(
            nn.Linear(num_feat, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes)
        )
    def forward(self, feat):
        return self.fc_classifier(feat)