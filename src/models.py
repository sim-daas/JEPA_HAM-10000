import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

class ProbeHead(nn.Module):
    def __init__(self, in_dim, num_classes=7, hidden=512, dropout=0.4, num_layers=2):
        super().__init__()
        
        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, num_classes))
        else:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden, hidden))
                layers.append(nn.BatchNorm1d(hidden))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
                
            layers.append(nn.Linear(hidden, num_classes))
            
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def build_vit_h14():
    return VisionTransformer(
        img_size=224, patch_size=14, embed_dim=1280, depth=32,
        num_heads=16, num_classes=0, global_pool="", class_token=False
    )

def load_ijepa_target_encoder(ckpt_path: str, device="cuda"):
    model = build_vit_h14()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = ckpt["target_encoder"]
    
    # Strip DDP / wrapper prefixes
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "").replace("backbone.", "")
        cleaned[new_k] = v
        
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"I-JEPA Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval().to(device)
    return model

class FullIJEPAModel(nn.Module):
    def __init__(self, ckpt_path: str, num_classes=7, hidden=512, dropout=0.4, num_layers=2):
        super().__init__()
        self.encoder = load_ijepa_target_encoder(ckpt_path, device="cpu")
        # Unfreeze encoder for full fine-tuning
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        self.encoder.train()
        
        self.head = ProbeHead(
            in_dim=1280,
            num_classes=num_classes,
            hidden=hidden,
            dropout=dropout,
            num_layers=num_layers
        )
        
    def forward(self, x):
        tokens = self.encoder(x)
        if tokens.dim() == 3:
            pooled = tokens.mean(dim=1)
        else:
            pooled = tokens
        return self.head(pooled)
