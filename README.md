# sat_emb
This is a repository to create image embeddings from a large satellite composite (XX enter link). The embeddings can be
used for a variety of tasks such as improving existing models for classification, as well as unsupervised tasks.

## Installation
Please install the conda environment called environment.yml. This will install all the necessary packages to run the code.

## Data download
Most of the data is not included in this repository. 
Please download the data from the following link: [link](www.includelinhere.com)

## Usage
### 1. Data preprocessing: create tiles
'''python create_tiles.py --size 9000000 --mask False --h3_shapes_path /data/vector/420_grid.parquet --vrt_file /data/raster/all_GHS-composite-S2.vrt --folder data/raster/england/unmasked/' '''




 
