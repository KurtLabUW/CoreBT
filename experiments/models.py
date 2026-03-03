from torch import nn
import torch


class LinearProbe(nn.Module):

    def __init__(self, embed_dim: int = 1536, num_classes: int = 10):
        super(LinearProbe, self).__init__()

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, embed_mri, embed_histo):
        # print(f'Inside forward: {embed_mri.size()}, {embed_histo.size()}')
        x = torch.cat([embed_mri, embed_histo], dim=1)
        # print(f'Inside forward: {x.size()}')
        return self.fc(x)

# class MultimodalClassifier(nn.Module):
#     def __init__(self, in_channels_rad: int, in_channels_histo: int,  hidden_dim: int = 512, num_classes: int = 10):
#         super().__init__()
#         self.in_channels_rad = int(in_channels_rad)
#         self.in_channels_histo = int(in_channels_histo)

#         self.hidden_dim = int(hidden_dim)
#         self.num_classes = num_classes

#         ## Adaptor Module
#         self.rad_embd = nn.Sequential(
#             nn.Linear(self.in_channels_rad, 512),
#             nn.GELU(),
#             nn.Linear(512, 512),
#             nn.LayerNorm(512),
#         )

#         self.path_embd = nn.Sequential(
#             nn.Linear(self.in_channels_histo, 512),
#             nn.GELU(),
#             nn.Linear(512, 512),
#             nn.LayerNorm(512),
#         )

#         self.retrieval_conv = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=512 * 2,
#                 out_channels=self.hidden_dim,
#                 kernel_size=3,
#                 padding=1,
#             ),
#             nn.GELU(),
#             nn.Conv1d(
#                 in_channels=self.hidden_dim,
#                 out_channels=self.hidden_dim,
#                 kernel_size=3,
#                 padding=1,
#             ),
#         )

#         self.prediction_head = nn.Sequential(
#             nn.Linear(self.hidden_dim, self.num_classes)
#         ) 


#     def forward(self, embed_mri, embed_histo):

#         radiology_feat = self.rad_embd(embed_mri)  ## 1x768 -> 1x512
#         pathology_feat = self.path_embd(embed_histo)  ## 1x768-> 1x512

#         ## Concatenate Radiology and Pathology Embeddings
#         feat = torch.cat([radiology_feat, pathology_feat.squeeze(1)], dim=1)  ## 1x1024

#         ## Fuse Multi-Modality Embeddings with 2-Layer Convolution
#         feat = self.retrieval_conv(torch.unsqueeze(feat, 2))

#         return self.prediction_head(feat.squeeze(2))




class MultimodalClassifier(nn.Module):
    def __init__(self, in_channels_rad, in_channels_histo,
                 hidden_dim=512, num_classes=10):
        super().__init__()

        self.rad_embd = nn.Sequential(
            nn.Linear(in_channels_rad, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(512),
        )

        self.path_embd = nn.Sequential(
            nn.Linear(in_channels_histo, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.LayerNorm(512),
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, embed_mri, embed_histo):

        rad = self.rad_embd(embed_mri)
        path = self.path_embd(embed_histo)

        # compute gating weights
        gate = self.gate(torch.cat([rad, path], dim=1))

        # gated residual fusion
        fused = rad + gate * path

        # print(
        #     f"Gate mean: {gate.mean().item():.4f} | "
        #     f"Gate std: {gate.std().item():.4f} | "
        #     f"Gate min: {gate.min().item():.4f} | "
        #     f"Gate max: {gate.max().item():.4f}"
        # )
        self.gateval = gate

        return self.classifier(fused)


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class ResidualFusionClassifier(nn.Module):
    def __init__(
        self,
        mri_probe_path: str,
        histo_probe_path: str,
        residual_hidden: int = 32,
        freeze_probes: bool = True,
        device: str = "cuda",
    ):
        super().__init__()

        # --------------------------------------------------
        # Load MRI probe
        # --------------------------------------------------
        mri_state = torch.load(mri_probe_path, weights_only=True, map_location=device)

        mri_weight = mri_state["fc.weight"]
        mri_bias = mri_state["fc.bias"]

        num_classes, mri_embed_dim = mri_weight.shape

        self.mri_probe = nn.Linear(mri_embed_dim, num_classes)
        self.mri_probe.weight.data.copy_(mri_weight)
        self.mri_probe.bias.data.copy_(mri_bias)

        # --------------------------------------------------
        # Load Histo probe
        # --------------------------------------------------
        histo_state = torch.load(histo_probe_path, weights_only=True, map_location=device)

        histo_weight = histo_state["fc.weight"]
        histo_bias = histo_state["fc.bias"]

        num_classes_histo, histo_embed_dim = histo_weight.shape

        assert (
            num_classes == num_classes_histo
        ), "MRI and Histo probes must have same number of classes"

        self.histo_probe = nn.Linear(histo_embed_dim, num_classes)
        self.histo_probe.weight.data.copy_(histo_weight)
        self.histo_probe.bias.data.copy_(histo_bias)

        self.num_classes = num_classes

        # --------------------------------------------------
        # Freeze probes (recommended for small dataset)
        # --------------------------------------------------
        if freeze_probes:
            for p in self.mri_probe.parameters():
                p.requires_grad = False
            for p in self.histo_probe.parameters():
                p.requires_grad = False

        # --------------------------------------------------
        # Small residual correction network
        # --------------------------------------------------
        self.residual = nn.Sequential(
            nn.Linear(2 * num_classes, residual_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(residual_hidden, num_classes),
        )

        # Learnable global scaling (start at strong prior)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        # self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, embed_mri, embed_histo):

        logits_mri = self.mri_probe(embed_mri)
        logits_histo = self.histo_probe(embed_histo)

        base_logits = self.alpha * logits_mri + self.beta * logits_histo

        residual_input = torch.cat([logits_mri, logits_histo], dim=1)
        delta = self.residual(residual_input)

        logits = base_logits + delta

        return logits