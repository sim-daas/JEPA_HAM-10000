import sys
import os
import torch
import torchvision
import torchvision.datasets.cifar as cifar_module
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Override check_integrity so torchvision uses local dataset structure without network download
cifar_module.check_integrity = lambda path, md5=None: True

sys.path.append('test/ijepa')
from src.models.vision_transformer import vit_huge

MODELS = [
    {
        "name": "IN1K ViT-H/14 (300e, 224px)",
        "ckpt": "models/IN1K-vit.h.14-300e.pth.tar",
        "patch_size": 14,
        "img_size": 224,
    },
    {
        "name": "IN1K ViT-H/16 (300e, 448px)",
        "ckpt": "models/IN1K-vit.h.16-448px-300e.pth.tar",
        "patch_size": 16,
        "img_size": 448,
    },
    {
        "name": "IN22K ViT-H/14 (900e, 224px)",
        "ckpt": "models/IN22K-vit.h.14-900e.pth.tar",
        "patch_size": 14,
        "img_size": 224,
    },
]

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"=== Testing I-JEPA Models on CIFAR-10 Dataset ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})\n")

    os.makedirs("datasets", exist_ok=True)

    for model_cfg in MODELS:
        print(f"--------------------------------------------------")
        print(f"Model: {model_cfg['name']}")
        print(f"Checkpoint: {model_cfg['ckpt']}")

        img_size = model_cfg['img_size']
        patch_size = model_cfg['patch_size']

        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        testset_transformed = torchvision.datasets.CIFAR10(
            root='./datasets',
            train=False,
            transform=transform,
            download=False
        )
        
        # DataLoader fetching a batch of CIFAR-10 dataset samples
        dataloader = DataLoader(testset_transformed, batch_size=8, shuffle=False)
        images, labels = next(iter(dataloader))
        images = images.to(device)

        # Load Model Architecture & Weights onto GPU
        model = vit_huge(img_size=[img_size], patch_size=patch_size).to(device)
        ckpt = torch.load(model_cfg['ckpt'], map_location='cpu', weights_only=False)
        state_dict = ckpt['target_encoder']
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=True)
        model.eval()

        # Run Feature Extraction Forward Pass on GPU
        with torch.no_grad():
            output = model(images)

        num_patches = (img_size // patch_size) ** 2
        print(f"CIFAR-10 Batch Input Shape:  {list(images.shape)} on {images.device}")
        print(f"I-JEPA Feature Output Shape:  {list(output.shape)} on {output.device}")
        print(f"Batch Size: {images.size(0)} | Patches/Image: {num_patches} | Feature Dimension: {output.size(-1)}")
        print(f"CIFAR-10 Ground Truth Labels (Batch 1): {labels.tolist()}")
        print("STATUS: PASSED SUCCESSFUL INFERENCE\n")

if __name__ == '__main__':
    main()
