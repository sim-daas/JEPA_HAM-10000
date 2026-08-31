import sys
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.append('test/ijepa')
from src.models.vision_transformer import vit_huge

MODELS = [
    {
        "name": "IN1K ViT-H/14 (300 epochs, 224px)",
        "ckpt": "models/IN1K-vit.h.14-300e.pth.tar",
        "patch_size": 14,
        "img_size": 224,
    },
    {
        "name": "IN1K ViT-H/16 (300 epochs, 448px)",
        "ckpt": "models/IN1K-vit.h.16-448px-300e.pth.tar",
        "patch_size": 16,
        "img_size": 448,
    },
    {
        "name": "IN22K ViT-H/14 (900 epochs, 224px)",
        "ckpt": "models/IN22K-vit.h.14-900e.pth.tar",
        "patch_size": 14,
        "img_size": 224,
    },
]

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"=== Running I-JEPA Model Tests ===")
    print(f"Target Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")
    print("-" * 50)

    # Prepare standard ImageNet preprocessing transform for input images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    for model_cfg in MODELS:
        print(f"\nTesting: {model_cfg['name']}")
        print(f"Path: {model_cfg['ckpt']}")
        
        if not os.path.exists(model_cfg['ckpt']):
            print(f"ERROR: Checkpoint file non-existent: {model_cfg['ckpt']}")
            continue

        # 1. Instantiate Model Architecture
        model = vit_huge(img_size=[model_cfg['img_size']], patch_size=model_cfg['patch_size'])
        model = model.to(device)

        # 2. Load Pretrained Checkpoint
        ckpt = torch.load(model_cfg['ckpt'], map_location='cpu', weights_only=False)
        state_dict = ckpt['target_encoder']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(new_state_dict, strict=True)
        model.eval()

        # 3. Create Sample Input Image
        # Resize input dynamically based on model specification (224px vs 448px)
        img_transform = transforms.Compose([
            transforms.Resize((model_cfg['img_size'], model_cfg['img_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Using a generated sample image tensor (simulating RGB image normalized to ImageNet stats)
        sample_img = Image.new('RGB', (model_cfg['img_size'], model_cfg['img_size']), color=(128, 128, 128))
        input_tensor = img_transform(sample_img).unsqueeze(0).to(device)

        # 4. Perform Forward Pass on GPU
        with torch.no_grad():
            output = model(input_tensor)

        num_patches = (model_cfg['img_size'] // model_cfg['patch_size']) ** 2
        print(f"STATUS: SUCCESS")
        print(f"Input Shape:  {list(input_tensor.shape)} (Device: {input_tensor.device})")
        print(f"Output Shape: {list(output.shape)} (Device: {output.device})")
        print(f"Expected Patches: {num_patches}, Output Embedding Dimension: {output.shape[-1]}")

if __name__ == '__main__':
    main()
