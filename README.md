# sat_emb
This is a repository to create image embeddings from a large [satellite composite](https://data.jrc.ec.europa.eu/dataset/0bd1dfab-e311-4046-8911-c54a8750df79) with a large pretrained transformer-based model ([Satlas](https://github.com/allenai/satlas/blob/main/SatlasPretrain.md)). The embeddings can be
used for a variety of tasks such as improving existing models for classification, as well as unsupervised tasks.

## Installation
Please install the conda environment called environment.yml. This will install all the necessary packages to run the code.

## Dependencies
The embeddings code can be run with Pytorch + CUDA or regular CPU depending on access to computing power.

## Data download 
Please download the data from the following link: [GoogleDrive](https://drive.google.com/drive/folders/1HJzoLHx9Bc5ZaOCl-GyzwGPBZzltPpii?usp=drive_link) (email me for access).
The Satlas model weights can be downloaded here: [Satlas Model Weights](https://huggingface.co/allenai/satlas-pretrain/blob/6e7d6eb1804162733c485a3f542fdc85a2addc55/satlas-model-v1-lowres.pth).

## Usage
### 1. Data preprocessing: create tiles
Create .tif files from the input .vrt file. The input .vrt file should contain the satellite composite. The files are saved in the output_folder. The h3_shapes_path should point to the h3 grid file. The grid420_path should point to the grid file. 
The size refers to the dataset size - e.g. number of tiles (and helps you create a smaller subset more easily). The mask parameter should be set to True if the tiles should be masked with the h3 grid. 
```
python create_tiles.py --size 9000000 --mask False --h3_shapes_path /data/vector/420_grid.parquet --grid420_path /data/vector/grid_complete.parquet --vrt_file /data/raster/all_GHS-composite-S2.vrt --folder /data/raster/england/unmasked/
```
Note: The whole England dataset includes 1,382,771 tiles. The whole dataset with clipped tiles is very large ~ about 700GB.

### 2. Create embeddings
Create embeddings by running the following command. The input folder should contain the tiles created in the previous step. The output file will contain the embeddings in a parquet file. The model weights path should point to the model weights file.
```
python create_embeddings_fpn_pool.py --input_folder /data/raster/england/unmasked/ --output_file /data/raster/england/unmasked/embeddings_england_pool_nomask.parquet --model_weights_path /data/model/satlas-model-v1-lowres.pth
```






 
