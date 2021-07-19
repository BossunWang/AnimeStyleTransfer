import argparse

import torch
import cv2
import numpy as np
import os

from model import Generator

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_image(image_path, x32=False):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    w = w // args.resize_scalar
    h = h // args.resize_scalar

    w = (w // 100) * 100
    h = (h // 100) * 100

    if x32: # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))
    else:
        img = cv2.resize(img, (w, h))

    img = torch.from_numpy(img)
    img = img / 127.5 - 1.0
    return img


def test(args):
    label_name_list = ['Hayao', 'Paprika', 'Shinkai']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = Generator()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "epoch" in checkpoint \
            and "generator" in checkpoint \
            and "discriminator" in checkpoint \
            and "G_optimizer" in checkpoint \
            and "D_optimizer" in checkpoint:
        generator.load_state_dict(checkpoint['generator'])

    generator.to(device).eval()
    print(f"model loaded: {args.checkpoint}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    for image_name in sorted(os.listdir(args.input_dir)):
        if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png", ".bmp", ".tiff"]:
            continue

        image = load_image(os.path.join(args.input_dir, image_name), args.x32)
        input = image.permute(2, 0, 1).unsqueeze(0).to(device)
        print(input.size())
        labels = np.arange(args.class_num)
        labels = labels.reshape(1, -1).repeat(input.size(0), axis=0)
        labels = torch.from_numpy(labels).to(device)
        labels = labels.long()

        for i in range(args.class_num):
            with torch.no_grad():
                out = generator(input, labels[:, i].view(-1), args.upsample_align).squeeze(0).permute(1, 2, 0).cpu().numpy()
                out = (out + 1)*127.5
                out = np.clip(out, 0, 255).astype(np.uint8)
            save_name = os.path.join(args.output_dir, image_name.replace('.jpg', '_' + label_name_list[i] + '.jpg'))
            cv2.imwrite(save_name, cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            print(f"image saved: {save_name}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='./pytorch_generator_Paprika.pt',
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='./samples/inputs',
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='./samples/results',
    )
    parser.add_argument(
        '--resize_scalar',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--upsample_align',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--x32',
        action="store_true",
    )
    parser.add_argument(
        '--class_num',
        type=int,
        default=3,
    )

    args = parser.parse_args()
    
    test(args)
