import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from peft import LoraConfig, get_peft_model


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total, 100 * trainable / total

class LoRAIJEPAModel(nn.Module):
    def __init__(self, ckpt_path: str, rank: int, num_classes=7, hidden=512, dropout=0.4, num_layers=2):
        super().__init__()
        self.encoder = load_ijepa_target_encoder(ckpt_path, device="cpu")
        
        # Apply LoRA using PEFT
        config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["qkv", "proj", "fc1", "fc2"],
            lora_dropout=0.1,
            bias="none"
        )
        self.encoder = get_peft_model(self.encoder, config)
        
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

class NaiveUnfreezeIJEPAModel(nn.Module):
    def __init__(self, ckpt_path: str, num_classes=7, hidden=512, dropout=0.4, num_layers=2):
        super().__init__()
        self.encoder = load_ijepa_target_encoder(ckpt_path, device="cpu")
        
        # 1. Freeze everything
        for p in self.encoder.parameters():
            p.requires_grad_(False)
            
        # 2. Unfreeze only the MLP of the last transformer block
        for p in self.encoder.blocks[-1].mlp.parameters():
            p.requires_grad_(True)
            
        # The head is naturally unfrozen upon initialization
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
