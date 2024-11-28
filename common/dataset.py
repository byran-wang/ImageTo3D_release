import os
import cv2
import numpy as np
import argparse


def data_preprocessed(image_in_f, image_out_f, crop_size, border_ratio):
    inpaint_rgba = cv2.imread(image_in_f, cv2.IMREAD_UNCHANGED)
    desired_size = int(crop_size * (1 - border_ratio))
    # Center the inpainted view
    mask = inpaint_rgba[:, :, 3]
    coords = np.nonzero(mask)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    h = y_max - y_min
    w = x_max - x_min
    scale = desired_size / max(h, w)
    h2 = int(h * scale)
    w2 = int(w * scale)
    y2_min = (crop_size - h2) // 2
    y2_max = y2_min + h2
    x2_min = (crop_size - w2) // 2
    x2_max = x2_min + w2
    # mask

    out_rgba_center = np.zeros((crop_size, crop_size, 4), dtype=np.uint8)
    out_rgba_center[y2_min:y2_max, x2_min:x2_max] = cv2.resize(inpaint_rgba[y_min:y_max, x_min:x_max], (w2, h2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(image_out_f, out_rgba_center)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_in_f", required=True, type=str, help="the image file after background removal")
    parser.add_argument("--image_out_f", required=True, type=str, help="the processed image file after centering and cropping")
    parser.add_argument("--crop_size", default=256, type=int, help="crop size")
    parser.add_argument("--border_ratio", default=0.1, type=float, help="border ratio")

    args = parser.parse_args()    
    data_preprocessed(args.image_in_f, args.image_out_f, args.crop_size, args.border_ratio)            