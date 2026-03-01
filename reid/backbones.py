from __future__ import division, absolute_import
import warnings
import torch
from torch import nn
import torchvision
import timm
__all__ = ['vit_b_16', 'dinov3_vit_b_16']

pretrained_urls = {
}

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

##########
# Network architecture
##########
class ViT(nn.Module):
    """ViT Network.
    """

    def __init__(
        self,
        num_classes,
        name,
        pretrained=True,
        feature_dim=768,
        loss='softmax',
        **kwargs
    ):
        super().__init__()

        # Create a pretrained vit model
        if name == 'vit_b_16':
            model = torchvision.models.vit_b_16(weights='IMAGENET1K_SWAG_E2E_V1')

        if name == 'dinov3_vit_b_16':
            model = timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=True)

        self.loss = loss
        self.feature_dim = feature_dim

        model.heads = Identity()

        self.model = model

        self.fc = self._construct_fc_layer(
            512, self.feature_dim, dropout_p=None
        )

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def featuremaps(self, x):
        x = self.model(x)

        if isinstance(x, (tuple, list)):
            x = x[0]

        if x.dim() == 3:
            x = x[:, 0]   # (B, C)

        return x

    def forward(self, x, return_featuremaps=False):
        x = self.featuremaps(x)

        if return_featuremaps:
            return x

        v = x
        v = v.view(v.size(0), -1)

        if self.fc is not None:
            v = self.fc(v)

        if not self.training:
            return v

        y = self.classifier(v)
        if self.loss == 'softmax':
            return y
        elif self.loss == 'triplet':
            return y, v
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


##########
# Instantiation
##########

def vit_b_16(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = ViT(
        num_classes,
        name='vit_b_16',
        pretrained=pretrained,
        feature_dim=768,
        loss=loss,
        **kwargs
    )
    return model

def dinov3_vit_b_16(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    model = ViT(
        num_classes,
        name='dinov3_vit_b_16',
        pretrained=pretrained,
        feature_dim=768,
        loss=loss,
        **kwargs
    )
    return model
