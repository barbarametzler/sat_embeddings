import argparse
import geopandas as gpd
import tools_satellite_demoland as tools

parser = argparse.ArgumentParser(description="Process satellite images.")
parser.add_argument("--size", type=int, default=20, help="Size of the subset.")
parser.add_argument("--mask", type=bool, default=True, help="Apply mask or not.")
parser.add_argument("--h3_shapes_path", type=str, required=True, help="Path to H3 shapes file.")
parser.add_argument("--grid420_path", type=str, required=True, help="Path to grid 420 file.")
parser.add_argument("--vrt_file", type=str, required=True, help="Path to VRT file.")
parser.add_argument("--folder", type=str, required=True, help="Path to folder.")

args = parser.parse_args()


def create_subset(size, mask, h3_shapes_path, grid420_path, vrt_file, folder):
    specs = {'chip_size': 420,
             'bands': [1, 2, 3],  # RGB
             'mosaic_p': (
                 vrt_file
             ),
             'folder': (
                 folder
             )
             }
    h3_shapes = gpd.read_parquet(h3_shapes_path)
    h3_shapes['hex_id'] = h3_shapes.index
    h3_shapes['centroid'] = h3_shapes['geometry'].centroid

    grid420 = gpd.read_parquet(grid420_path)
    ch420 = gpd.sjoin(grid420, h3_shapes, how='left', op='contains')

    if mask == False:
        print('mask is false')
        df = ch420[0:size].copy()
        centroid = df.centroid
        df['X'] = centroid.x.astype(int)
        df['Y'] = centroid.y.astype(int)
        tools.spilled_bag_of_chips(df, specs, npartitions=16)

    if mask == True:
        print('mask is true')
        dff = h3_shapes[0:size].copy()
        centroidd = dff.centroid
        dff['X'] = centroidd.x.astype(int)
        dff['Y'] = centroidd.y.astype(int)
        tools.spilled_bag_of_chips(dff, specs, npartitions=16)

create_subset(args.size, args.mask, args.h3_shapes_path, args.grid420_path, args.vrt_file, args.folder)
