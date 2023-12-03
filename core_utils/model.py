import torch
import torch.nn as nn


# derived motivation from CLAM attention branches when building the attention network
class AttentionNetwork(nn.Module):
    def __init__(self, L=128, D=64, dropout=False, n_classes=1):
        super(AttentionNetwork, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.ReLU()]

        self.attention_b = [
            nn.Linear(L, D),
            nn.Sigmoid()]

        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        out = a.mul(b)
        out = self.attention_c(out)
        out = self.softmax(out)
        return out


class CNN(nn.Module):
    def __init__(self, num_classes=2, channel_depth=128, group_norm=None):
        super(CNN, self).__init__()

        # >>> input = torch.randn(20, 128, 10, 10)
        # >>> # Separate 128 channels into 3 groups
        # >>> m = nn.GroupNorm(3, 128)
        # >>> # Separate 128 channels into 128 groups (equivalent with InstanceNorm)
        # >>> m = nn.GroupNorm(128, 128)
        # >>> # Put all 128 channels into a single group (equivalent with LayerNorm)
        # >>> m = nn.GroupNorm(1, 128)

        if group_norm is None:
            group_norm = [1, 128]

        self.layer1 = nn.Sequential(
            nn.Conv2d(channel_depth, channel_depth, (3, 3), padding=1, stride=2),
            nn.GroupNorm(group_norm[0], group_norm[1]),
            nn.ReLU(inplace=True),
        )

        # Convolutional layers
        self.layer2 = nn.Sequential(
            nn.Conv2d(channel_depth, channel_depth, (3, 3), padding=1, stride=2),
            nn.GroupNorm(group_norm[0], group_norm[1]),
            nn.ReLU(inplace=True),
        )

        # Convolutional layers
        self.layer3 = nn.Sequential(
            nn.Conv2d(channel_depth, channel_depth, (3, 3), padding=1),
            nn.GroupNorm(group_norm[0], group_norm[1]),
            nn.ReLU(inplace=True),
        )

        # Convolutional layers
        self.layer4 = nn.Sequential(
            nn.Conv2d(channel_depth, channel_depth, (3, 3), padding=1),
            nn.GroupNorm(group_norm[0], group_norm[1]),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.MaxPool2d((2, 2))

        self.fc1 = nn.Linear(channel_depth, num_classes)

        self.drop_out = nn.Dropout(0.2)

    def forward(self, x, return_intermediates=False):
        out_l1 = self.layer1(x)
        out_l2 = self.layer2(out_l1)
        out_m1 = self.maxpool(out_l2)
        out_l3 = self.layer3(out_m1)
        out_l4 = self.layer4(out_l3)
        out_m2 = self.maxpool(out_l4)
        out = self.avgpool(out_m2)
        out_flat = torch.flatten(out, 1)
        out = self.fc1(out_flat)

        if return_intermediates:
            return out_l1, out_l2, out_l3, out_l4, out_flat, out

        return out


def zerooneeps(scores, eps=1e-5, dim=-1):
    scores_min = scores.min(axis=dim, keepdim=True)[0]
    scores_max = scores.max(axis=dim, keepdim=True)[0]
    return (scores - scores_min) / (scores_max - scores_min + eps)


class ANSAC(nn.Module):
    def __init__(self, group_norm, num_classes=2, dropout=False, size=None, normalize=False):
        super(ANSAC, self).__init__()

        if size is None:
            size = [576, 256]

        self.normalize = normalize
        self.classifier = CNN(num_classes=num_classes, group_norm=group_norm)

        self.attention_net_background = AttentionNetwork(L=size[0], D=size[1], n_classes=1, dropout=dropout)
        self.attention_net_tumor = AttentionNetwork(L=size[0], D=size[1], n_classes=1, dropout=dropout)
        self.attention_net_stroma = AttentionNetwork(L=size[0], D=size[1], n_classes=1, dropout=dropout)
        self.attention_net_lymphocyte = AttentionNetwork(L=size[0], D=size[1], n_classes=1, dropout=dropout)
        self.attention_net_necrosis = AttentionNetwork(L=size[0], D=size[1], n_classes=1, dropout=dropout)
        self.attention_net_other = AttentionNetwork(L=size[0], D=size[1], n_classes=1, dropout=dropout)

    def forward(self, x, attention_only=False, output=False):
        features, weights, device = x

        features = features.to(device, dtype=torch.float)
        weights = weights.to(device, dtype=torch.float)

        weights = weights.reshape(weights.shape[0], 6400, 576, 6).permute(0, 1, 3, 2)
        weights = zerooneeps(weights, eps=1e-5, dim=2)

        attn_weights_bg = self.attention_net_background(weights[:, :, 0, :].to(device, dtype=torch.float))
        attn_weights_tumor = self.attention_net_tumor(weights[:, :, 1, :].to(device, dtype=torch.float))
        attn_weights_stroma = self.attention_net_stroma(weights[:, :, 2, :].to(device, dtype=torch.float))
        attn_weights_lym = self.attention_net_lymphocyte(weights[:, :, 3, :].to(device, dtype=torch.float))
        attn_weights_nec = self.attention_net_necrosis(weights[:, :, 4, :].to(device, dtype=torch.float))
        attn_weights_other = self.attention_net_other(weights[:, :, 5, :].to(device, dtype=torch.float))

        attn_weights = torch.cat(
            [attn_weights_bg, attn_weights_tumor, attn_weights_stroma, attn_weights_lym, attn_weights_nec,
             attn_weights_other], dim=-1)
        attn_weights = torch.sum(attn_weights, dim=-1).unsqueeze(2)

        if attention_only:
            return attn_weights

        if self.normalize:
            attn_weights = zerooneeps(attn_weights, eps=1e-5, dim=1)

        attn_weights = attn_weights.reshape(attn_weights.shape[0], 80, 80).to(device, dtype=torch.float)

        features_weighted = torch.einsum('bijk, bjk -> bijk', features, attn_weights)

        out = self.classifier(features_weighted)

        if output:
            _, _, _, _, outf, out2 = self.classifier(features_weighted, return_intermediates=True)
            return out2, attn_weights, features_weighted, outf

        return out, attn_weights
