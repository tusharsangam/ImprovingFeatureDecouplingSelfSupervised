import torch
import torch.nn as nn


#keeping same as author https://github.com/philiptheother/FeatureDecoupling/tree/master/pytorch_feature_decoupling
class AlexNetClassifier(nn.Module):
    def __init__(self, num_classes=4, num_feat=2048):
        super(AlexNetClassifier, self).__init__()
        self.fc_classifier = nn.Sequential(
            nn.Linear(num_feat, num_classes)
        )
    def forward(self, feat):
        return self.fc_classifier(feat)