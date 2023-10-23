""" Define the Hubert model
"""


from transformers import HubertModel
from torch import nn
import torch.nn.functional as F

class HubertForAudioClassification(nn.Module):
    def __init__(self, adapter_hidden_size = 64):
        super().__init__()

        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")

        hidden_size = self.hubert.config.hidden_size

        self.adaptor = nn.Sequential(
            nn.Linear(hidden_size, adapter_hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.05),
            nn.Linear(adapter_hidden_size, hidden_size),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, adapter_hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.05),
            nn.Linear(adapter_hidden_size, 36), # 36 classes
        )

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder 
        so that its parameter will not be updated during training.
        """
        self.hubert.feature_extractor._freeze_parameters()

    def forward(self, x):
        # x shape: (B,E)
        x = self.hubert(x).last_hidden_state
        x = F.layer_norm(x, x.shape[1:])
        x = self.adaptor(x)
        # pooling
        x, _ = x.max(dim=1)

        # Mutilayer perceptron with log softmax for classification
        out = self.classifier(x)

        # Remove last dimension
        return out
        # return shape: (B, total_labels)
