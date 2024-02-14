import os
import glob
import torch
import torchvision
import argparse
import pandas as pd
import numpy as np
import tifffile

import collections
from torchvision.ops import FeaturePyramidNetwork

#num_workers =0


def load_model(model_weights_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.swin_transformer.swin_v2_b().to(device)
    full_state_dict = torch.load(model_weights_path, map_location=device)

    # Extract just the Swin backbone parameters from the full state dict.
    swin_prefix = 'backbone.backbone.'
    fpn_prefix = 'intermediates.0.fpn.'  # FPN

    swin_state_dict = {k[len(swin_prefix):]: v for k, v in full_state_dict.items() if k.startswith(swin_prefix)}
    model.load_state_dict(swin_state_dict)

    fpn_state_dict = {k[len(fpn_prefix):]: v for k, v in full_state_dict.items() if k.startswith(fpn_prefix)}
    fpn = FeaturePyramidNetwork([128, 256, 512, 1024], out_channels=128).to(device)
    fpn.load_state_dict(fpn_state_dict)

    return model, fpn


def return_featurevector(image_file, model, fpn):
    # Load TIF image using tifffile
    im = tifffile.imread(image_file)

    # Assuming the TIF image is already processed from B04 (red), B03 (green), and B02 (blue) bands
    # Normalize the pixel values to the range [0, 1]
    im = im.astype(float) / 255.0

    # Transpose the image array to match the expected format
    im_ = im.transpose(2, 0, 1)

    # Convert NumPy array to PyTorch tensor
    xx = torch.from_numpy(im_).float() #.detach().cpu().numpy().

    if torch.cuda.is_available():
        # Move your tensor to GPU
        xx = xx.to('cuda')
    x = xx[None, :, :, :]
    #x = x.to('cuda')
    outputs = []

    # Pass the image through the model layers
    for layer in model.features:
        x = layer(x)
        outputs.append(x.permute(0, 3, 1, 2))
    map1, map2, map3, map4 = outputs[-7], outputs[-5], outputs[-3], outputs[-1]

    # Process feature maps with FPN and extract features
    feature_maps_raw = [map1, map2, map3, map4]
    inp = collections.OrderedDict([('feat{}'.format(i), el) for i, el in enumerate(feature_maps_raw)])
    output = fpn(inp)
    output = list(output.values())

    avgpool = torch.nn.AdaptiveAvgPool2d(1)
    features = avgpool(output[-1])[:, :, 0, 0]

    return features.detach().cpu().numpy()

def process_images_in_folder_with_index(folder_path, model, fpn):
    feature_vectors_with_index = []

    # Iterate over files in the main folder
    for index, filename in enumerate(os.listdir(folder_path)):
        image_path = os.path.join(folder_path, filename)

        # Check if it's an image file (e.g., .tif)
        if filename.endswith(".tif"):  # Adjust the file extension if needed
            # Create feature vector for the current image
            feature_vector = return_featurevector(image_path, model, fpn)

            # Append the feature vector along with the file index to the list
            feature_vectors_with_index.append((index, feature_vector))

    return feature_vectors_with_index

def feature_vector_with_index(folder_path, model, fpn):
    p = glob.glob(folder_path+'*.tif')
    feature_vectors = [return_featurevector(file, model, fpn) for file in p]
    print('Creating feature vectors...')
    df_fe = pd.DataFrame(np.concatenate(feature_vectors, axis=0).squeeze())
    print('Creating DataFrame...')
    return df_fe

def main(input_folder, output_file, model_weights_path):
    model, fpn = load_model(model_weights_path)
    emb_masked = feature_vector_with_index(input_folder, model, fpn)
    emb_masked.to_csv(output_file)
    print('Saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process satellite images and extract feature vectors.")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input folder containing TIF images.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output CSV file.")
    parser.add_argument("--model_weights_path", type=str, required=True, help="Path to the model weights file.")
    args = parser.parse_args()

    main(args.input_folder, args.output_file, args.model_weights_path)


#def main():
#emb_masked = feature_vector_with_index('../sat_demo/data/subset/summed_masked/*/*.tif')
#emb_masked.to_csv('../sat_demo/data/NC_emb_sub_satlas_masked_pooled.csv')
#print('saved')
