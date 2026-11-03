import os
import argparse
import yaml
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from photocast_dataloader import get_photocast_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yml",
                        help='Config file with the photocast data settings')
    parser.add_argument('--output_path', type=str, default='photocast-train-256',
                        help='Folder where transformed images will be saved')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for the random panorama crops (makes the reference set reproducible)')
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fid:
        config = yaml.safe_load(fid)

    # Same dataset and transforms as training (random panorama crops, normalized to [-1, 1])
    dataset_train = get_photocast_dataloader(config)
    print(dataset_train)

    # Seeding before creating the loader also fixes the worker seeds, so the crops are reproducible
    torch.manual_seed(args.seed)

    data_loader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False
    )

    os.makedirs(args.output_path, exist_ok=True)

    to_pil = transforms.ToPILImage()
    global_idx = 0

    from tqdm import tqdm
    for batch in tqdm(data_loader):
        # undo normalization: [-1, 1] -> uint8 [0, 255]
        batch_images = ((batch["tensors"][0] + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
        for i in range(batch_images.size(0)):
            img_tensor = batch_images[i]

            pil_img = to_pil(img_tensor)
            out_path = os.path.join(
                args.output_path,
                f"transformed_{global_idx:08d}.png"
            )
            pil_img.save(out_path, format='PNG', compress_level=0)
            global_idx += 1

    print(f"Finished saving {global_idx} images.")


if __name__ == "__main__":
    main()
