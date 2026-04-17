from torch import nn
import torch
import torch.nn.functional as F

class LinearProbe(nn.Module):

    def __init__(self, embed_dim: int = 1536, num_classes: int = 10):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, embed_mri, embed_histo):
        x = torch.cat([embed_mri, embed_histo], dim=1)
        return self.fc(x)

class ResidualFusionClassifier(nn.Module):
    def __init__(self, mri_probe_path, histo_probe_path,
                 residual_hidden=32, freeze_probes=True, device="cuda"):

        super().__init__()

        # Load MRI probe
        mri_state = torch.load(mri_probe_path, map_location=device)
        num_classes = mri_state["fc.bias"].shape[0]
        in_dim_mri = mri_state["fc.weight"].shape[1]

        self.mri_probe = nn.Linear(in_dim_mri, num_classes)
        self.mri_probe.weight.data = mri_state["fc.weight"]
        self.mri_probe.bias.data = mri_state["fc.bias"]

        # Load Histo probe
        histo_state = torch.load(histo_probe_path, map_location=device)
        in_dim_histo = histo_state["fc.weight"].shape[1]

        self.histo_probe = nn.Linear(in_dim_histo, num_classes)
        self.histo_probe.weight.data = histo_state["fc.weight"]
        self.histo_probe.bias.data = histo_state["fc.bias"]

        if freeze_probes:
            for p in self.mri_probe.parameters():
                p.requires_grad = False
            for p in self.histo_probe.parameters():
                p.requires_grad = False
     
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))

        self.residual = nn.Sequential(
            nn.Linear(2 * num_classes, residual_hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(residual_hidden, num_classes)
        )

    def forward(self, embed_mri, embed_histo, mask):

        logits_mri = self.mri_probe(embed_mri)
        logits_histo = self.histo_probe(embed_histo)

        mri_mask = mask[:, 0].unsqueeze(1)
        histo_mask = mask[:, 1].unsqueeze(1)

        logits_mri = logits_mri * mri_mask
        logits_histo = logits_histo * histo_mask

        denom = torch.clamp(mri_mask + histo_mask, min=1.0)

        base = (self.alpha * logits_mri +
                self.beta * logits_histo) / denom

        delta = self.residual(torch.cat([logits_mri, logits_histo], dim=1))

        return base + delta




class AdditiveGatedFusion(nn.Module):
    def __init__(self, mri_probe_path, histo_probe_path, 
                 bottleneck_dim=16, freeze_probes=True, device="cuda"):
        super().__init__()

        # --- 1. Load Probes (Fixed Anchors) ---
        mri_state = torch.load(mri_probe_path, map_location=device)
        num_classes = mri_state["fc.bias"].shape[0]
        in_dim_mri = mri_state["fc.weight"].shape[1]
        
        self.mri_probe = nn.Linear(in_dim_mri, num_classes)
        self.mri_probe.load_state_dict({"weight": mri_state["fc.weight"], "bias": mri_state["fc.bias"]})

        histo_state = torch.load(histo_probe_path, map_location=device)
        in_dim_histo = histo_state["fc.weight"].shape[1]
        
        self.histo_probe = nn.Linear(in_dim_histo, num_classes)
        self.histo_probe.load_state_dict({"weight": histo_state["fc.weight"], "bias": histo_state["fc.bias"]})

        if freeze_probes:
            for p in self.parameters(): p.requires_grad = False

        self.mri_proj = nn.Linear(in_dim_mri, bottleneck_dim)
        self.histo_proj = nn.Linear(in_dim_histo, bottleneck_dim)
        
        self.residual_net = nn.Sequential(
            nn.BatchNorm1d(bottleneck_dim * 2),
            nn.Dropout(0.5), # High dropout for small data
            nn.Linear(bottleneck_dim * 2, num_classes)
        )
        self.res_scale = nn.Parameter(torch.tensor([0.01]))

    def forward(self, embed_mri, embed_histo, mask):
        mri_mask = mask[:, 0].unsqueeze(1)
        histo_mask = mask[:, 1].unsqueeze(1)

        logits_mri = self.mri_probe(embed_mri) * mri_mask
        logits_histo = self.histo_probe(embed_histo) * histo_mask
        
        # Simple mean of available probes
        count = torch.clamp(mri_mask + histo_mask, min=1e-6)
        base_logits = (logits_mri + logits_histo) / count

        mri_low = self.mri_proj(embed_mri) * mri_mask
        histo_low = self.histo_proj(embed_histo) * histo_mask
        
        feat_fusion = torch.cat([mri_low, histo_low], dim=1)
        delta = self.residual_net(feat_fusion)

        return base_logits + (self.res_scale * delta)

class LatentAdditiveFusion(nn.Module):
    def __init__(self, mri_probe_path, histo_probe_path, 
                 latent_dim=16, device="cuda"):
        super().__init__()

        # --- 1. Load Probes (Anchors) ---
        mri_state = torch.load(mri_probe_path, map_location=device)
        num_classes = mri_state["fc.bias"].shape[0]
        in_dim_mri = mri_state["fc.weight"].shape[1]
        
        self.mri_probe = nn.Linear(in_dim_mri, num_classes)
        self.mri_probe.load_state_dict({"weight": mri_state["fc.weight"], "bias": mri_state["fc.bias"]})

        histo_state = torch.load(histo_probe_path, map_location=device)
        in_dim_histo = histo_state["fc.weight"].shape[1]
        
        self.histo_probe = nn.Linear(in_dim_histo, num_classes)
        self.histo_probe.load_state_dict({"weight": histo_state["fc.weight"], "bias": histo_state["fc.bias"]})

        # Freeze them to keep that high starting F1
        for p in self.parameters(): p.requires_grad = False

        self.shared_proj = nn.Sequential(
            nn.Linear(in_dim_mri + in_dim_histo, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )

        self.residual_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(latent_dim, num_classes)
        )

        self.res_gate = nn.Parameter(torch.tensor(-4.0))

    def forward(self, embed_mri, embed_histo, mask):
        mri_mask = mask[:, 0].unsqueeze(1)
        histo_mask = mask[:, 1].unsqueeze(1)

        l_mri = self.mri_probe(embed_mri) * mri_mask
        l_histo = self.histo_probe(embed_histo) * histo_mask
        
        base_logits = (l_mri + l_histo) / (mri_mask + histo_mask + 1e-8)

        m_feat = embed_mri * mri_mask
        h_feat = embed_histo * histo_mask
        
        combined = torch.cat([m_feat, h_feat], dim=1)
        latent_features = self.shared_proj(combined)
        

        delta = self.residual_head(latent_features)
        
        gate = torch.sigmoid(self.res_gate)
        return base_logits + (gate * delta)



class GatedResidualFusion(nn.Module):
    def __init__(self, mri_probe_path, histo_probe_path,
                 residual_hidden=32, freeze_probes=True, device="cuda"):
        super().__init__()

        mri_state = torch.load(mri_probe_path, map_location=device)
        num_classes = mri_state["fc.bias"].shape[0]
        in_dim_mri = mri_state["fc.weight"].shape[1]
        self.mri_probe = nn.Linear(in_dim_mri, num_classes)
        self.mri_probe.load_state_dict({"weight": mri_state["fc.weight"], "bias": mri_state["fc.bias"]})

        histo_state = torch.load(histo_probe_path, map_location=device)
        in_dim_histo = histo_state["fc.weight"].shape[1]
        self.histo_probe = nn.Linear(in_dim_histo, num_classes)
        self.histo_probe.load_state_dict({"weight": histo_state["fc.weight"], "bias": histo_state["fc.bias"]})

        if freeze_probes:
            for p in self.mri_probe.parameters(): p.requires_grad = False
            for p in self.histo_probe.parameters(): p.requires_grad = False


        self.modality_gate = nn.Parameter(torch.tensor([0.0, 0.0])) 

        self.shared_proj = nn.Sequential(
            nn.Linear(in_dim_mri + in_dim_histo, residual_hidden),
            nn.LayerNorm(residual_hidden),
            nn.GELU(),
            nn.Dropout(0.5) 
        )

        self.residual_head = nn.Linear(residual_hidden, num_classes)

        self.residual_gate = nn.Parameter(torch.tensor(-4.0))

    def forward(self, embed_mri, embed_histo, mask):
        mri_mask = mask[:, 0].unsqueeze(1)    # [B, 1]
        histo_mask = mask[:, 1].unsqueeze(1)  # [B, 1]

        logits_mri = self.mri_probe(embed_mri)
        logits_histo = self.histo_probe(embed_histo)

        weights = F.softmax(self.modality_gate, dim=0)
        
        mri_part = weights[0] * logits_mri * mri_mask
        histo_part = weights[1] * logits_histo * histo_mask

        denom = (weights[0] * mri_mask + weights[1] * histo_mask) + 1e-8
        base_logits = (mri_part + histo_part) / denom

        combined_feats = torch.cat([embed_mri * mri_mask, embed_histo * histo_mask], dim=1)
        latent = self.shared_proj(combined_feats)
        delta = self.residual_head(latent)

        res_weight = torch.sigmoid(self.residual_gate)
        
        return base_logits + (res_weight * delta)



class RobustGatedFusion(nn.Module):
    def __init__(self, mri_probe_path, histo_probe_path, 
                 residual_hidden=64, freeze_probes=True, device="cuda"):
        super().__init__()

        mri_state = torch.load(mri_probe_path, map_location=device)
        num_classes = mri_state["fc.bias"].shape[0]
        in_dim_mri = mri_state["fc.weight"].shape[1]
        
        self.mri_probe = nn.Linear(in_dim_mri, num_classes)
        self.mri_probe.load_state_dict({"weight": mri_state["fc.weight"], "bias": mri_state["fc.bias"]})

        histo_state = torch.load(histo_probe_path, map_location=device)
        in_dim_histo = histo_state["fc.weight"].shape[1]
        
        self.histo_probe = nn.Linear(in_dim_histo, num_classes)
        self.histo_probe.load_state_dict({"weight": histo_state["fc.weight"], "bias": histo_state["fc.bias"]})

        if freeze_probes:
            for p in self.mri_probe.parameters(): p.requires_grad = False
            for p in self.histo_probe.parameters(): p.requires_grad = False

        self.mri_norm = nn.LayerNorm(in_dim_mri)
        self.histo_norm = nn.LayerNorm(in_dim_histo)

        self.gate_generator = nn.Sequential(
            nn.Linear(in_dim_mri + in_dim_histo, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

        self.fusion_path = nn.Sequential(
            nn.Linear(in_dim_mri + in_dim_histo, residual_hidden),
            nn.LayerNorm(residual_hidden),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(residual_hidden, num_classes)
        )

        nn.init.zeros_(self.fusion_path[-1].weight)
        nn.init.zeros_(self.fusion_path[-1].bias)

        self.res_gate = nn.Parameter(torch.tensor([-4.0]))

    def forward(self, embed_mri, embed_histo, mask):
        """
        mask: Tensor of [Batch, 2] where 1 is present, 0 is missing.
        """
        mri_mask = mask[:, 0:1]
        histo_mask = mask[:, 1:2]

        logits_mri = self.mri_probe(embed_mri)
        logits_histo = self.histo_probe(embed_histo)

        m_feat = self.mri_norm(embed_mri) * mri_mask
        h_feat = self.histo_norm(embed_histo) * histo_mask
        combined_feats = torch.cat([m_feat, h_feat], dim=1)

        gate_logits = self.gate_generator(combined_feats)
        gate_logits = gate_logits.masked_fill(mask == 0, -1e9)
        gate_weights = F.softmax(gate_logits, dim=1)

        base_logits = (gate_weights[:, 0:1] * logits_mri) + \
                      (gate_weights[:, 1:2] * logits_histo)

        res_delta = self.fusion_path(combined_feats)
        res_scale = torch.sigmoid(self.res_gate)

        return base_logits + (res_scale * res_delta)

