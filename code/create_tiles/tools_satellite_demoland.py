import os
import pandas
import rasterio
import rasterio.mask
import geopandas
import numpy as np
import dask.bag as dbag
from numba import njit


def bag_of_chips(chip_bbs, specs, npartitions):
    '''
    Load imagery for `chip_bbs` using a Dask bag
    ...
    
    Arguments
    ---------
    chip_bbs : GeoDataFrame
               Geo-table with bounding boxes of the chips to load
    specs : dict
            Metadata dict, including, at least:
            - `bands`: band index of each band of interest
            - `chip_size`: size of each chip size expressed in pixels
            - `mosaic_p`: path to the mosaic/file of imagery
    npartitions : int
                  No. of partitions to split `chip_bbs` before sending to
                  Dask for distributed computation
    Returns
    -------
    chips : ndarray
            Numpy tensor of (N, chip_size, chip_size, n_bands) dimension 
            with imagery data   
    '''
    # Split chip_bbs
    thr = np.linspace(0, chip_bbs.shape[0], npartitions+1, dtype=int)
    chunks = [
        (chip_bbs.iloc[thr[i]:thr[i+1], :], specs) for i in range(len(thr)-1)
    ]
    # Set up the bag
    bag = dbag.from_sequence(
        chunks, npartitions=npartitions
    ).map(chip_loader)
    # Compute
    chips = np.concatenate(bag.compute())
    return chips


def chip_loader(pars):
    '''
    Load imagery for `chip_bbs`
    ...
    
    Arguments (wrapped in `pars`)
    -----------------------------
    chip_bbs : GeoDataFrame
               Geo-table with bounding boxes of the chips to load
    specs : dict
            Metadata dict, including, at least:
            - `bands`: band index of each band of interest
            - `chip_size`: size of each chip size expressed in pixels
            - `mosaic_p`: path to the mosaic/file of imagery
    Returns
    -------
    chips : ndarray
            Numpy tensor of (N, chip_size, chip_size, n_bands) dimension 
            with imagery data
    '''
    chip_bbs, specs = pars
    b = len(specs['bands'])
    s = specs['chip_size']
    chips = np.zeros((chip_bbs.shape[0], b, s, s))
    with rasterio.open(specs['mosaic_p']) as src:
        for i, tup in enumerate(chip_bbs.itertuples()):
            img, transform = rasterio.mask.mask(
                src, [tup.geometry], crop=True
            )
            chips[i, :, :, :] = img[:b, :s, :s]
    chips = np.moveaxis(chips, 1, -1)
    return chips


def spilled_bag_of_chips(chip_bbs, specs, npartitions):
    '''
    Load imagery for `chip_bbs` using a Dask bag and spill then to disk
    ...
    
    Arguments
    ---------
    chip_bbs : GeoDataFrame
               Geo-table with bounding boxes of the chips to load
    specs : dict
            Metadata dict, including, at least:
            - `bands`: band index of each band of interest
            - `chip_size`: size of each chip size expressed in pixels
            - `mosaic_p`: path to the mosaic/file of imagery
    npartitions : int
                  No. of partitions to split `chip_bbs` before sending to
                  Dask for distributed computation
 
    '''
    # Split chip_bbs
    thr = np.linspace(0, chip_bbs.shape[0], npartitions+1, dtype=int)
    chunks = [
        (chip_bbs.iloc[thr[i]:thr[i+1], :], specs) for i in range(len(thr)-1)
    ]
    #print(chunks)  # Add this line to print the chunks before creating the bag

    # create folders
    os.makedirs(f"{specs['folder']}", exist_ok=True)
    # Set up the bag
    bag = dbag.from_sequence(
        chunks, npartitions=npartitions
    ).map(spill_chips_to_disk)
    
    bag.compute()

    
def spill_chips_to_disk(pars):
    '''
    Get individual chips and spill then to disk, one chip per file.
    '''

    chip_bbs, specs = pars
    b = len(specs['bands'])
    s = specs['chip_size']
    
    with rasterio.open(specs['mosaic_p']) as src:
        profile = src.profile
        profile.update(
            count=b,
            width=s,
            height=s,
            tiled=False,
            driver="GTiff",
            dtype=rasterio.uint8,
        )
        for tup in chip_bbs.itertuples():
            img, transform = rasterio.mask.mask(
                src, [tup.geometry], crop=True, all_touched=True
            )
            img = img[:b, :s, :s]
            for i, (l_min, l_max) in enumerate([(350, 1600), (500, 1600), (600, 1800)]):
                img[i][img[i] > l_max] = l_max
                img[i][img[i] < l_min] = l_min
                a_std = (img[i] - l_min) / (l_max - l_min)
                img[i] = a_std * 255
            path = f"{specs['folder']}/{tup.X}_{tup.Y}.tif"
            with rasterio.open(path, 'w', **profile) as dst:
                dst.write(img[:b, :s, :s].astype(rasterio.uint8))
            
            
            

def dask_map_seq(f, items, client, njobs=None):
    '''
    Execute a queue of jobs in Dask with `njobs` in parallel
    ...
    
    Arguments
    ---------
    f : method
        Function to apply
    items : sequence
            List of items to apply `f` to
    client : dask.distributed.Client
             Existing Dask client to run the jobs on
    njobs : None/int
            Number of parallel jobs to run at any given time. If
            None then it matches the number of workers in the `client`
    
    Returns
    -------
    None
    '''
    from dask.distributed import as_completed
    iitems = iter(items)
    if njobs is None:
        njobs = len(client.cluster.workers)
    first_futures = [
        client.submit(f, next(iitems)) for i in range(njobs)
    ]
    ac = as_completed(first_futures)
    for finished_future in ac:
        try:
            new_future = client.submit(f, next(iitems))
            ac.add(new_future)
        except StopIteration:
            pass
    return None


def build_grid(x_coords, y_coords, chip_res, crs=None):
    '''
    Build a grid of chips from a raster with (`nx_pixels`,
    `ny_pixels`) of dimension at `chip_res` resolution
    ...
    
    Arguments
    ---------
    x_coords : xarray.DataArray
               Horizontal coordinates of pixel locations
    y_coords : xarray.DataArray
               Vertical coordinates of pixel locations
    chip_res : int
               Size of the chip in number of pixels
    crs : None/str
          [Optional. Default=None] CRS in which `x_coords` and
          `y_coords` are expressed in
    
    Returns
    -------
    grid : GeoDataFrame
    '''
    chip_xys, chip_len = coords2xys(
        x_coords, y_coords, chip_res
    )
    grid = geopandas.GeoSeries(
        geopandas.points_from_xy(
            chip_xys[:, 0], chip_xys[:, 1]
        ),
        crs=crs
    ).buffer(chip_len/2, cap_style=3)
    grid = geopandas.GeoDataFrame(
        {
            'geometry': grid,
            'X': chip_xys[:, 0],
            'Y': chip_xys[:, 1]
        }, crs=crs
    )
    return grid


def coords2xys(x_coords, y_coords, chip_res):
    '''
    Build set of centroid coordinates of the chip grid resulting from
    a cartesian product of `x_coords` and `y_coords`, with an
    aggregation level of `chip_res` pixels
    ...
    
    x_coords : xarray.DataArray
               Horizontal coordinates of pixel locations
    y_coords : xarray.DataArray
               Vertical coordinates of pixel locations
    chip_res : int
               Size of the chip in number of pixels
               
    Returns
    -------
    chip_xys : ndarray
               Nx2 array with XY coordinates for the centroid of 
               each pixel in the chip grid
    chip_len : float
               Length of the chip side expressed in the same units
               as `x_coords` and `y_coords`
    '''
    chip_xs = x_coords.coarsen(
        {x_coords.name: chip_res}, boundary='trim'
    ).mean()
    chip_ys = y_coords.coarsen(
        {y_coords.name: chip_res}, boundary='trim'
    ).mean()
    chip_len = float(chip_xs[1] - chip_xs[0])
    chip_xys = cartesian((chip_xs.values, chip_ys.values))
    return chip_xys, chip_len


def coords2xys_parquet(
    out_p, npartitions, x_coords, y_coords, chip_res
):
    '''
    Build set of centroid coordinates of the chip grid resulting from
    a cartesian product of `x_coords` and `y_coords`, with an
    aggregation level of `chip_res` pixels
    
    Output coordinates are directly written into a set of Parquet
    files
    ...
    
    out_p : str
            Path to the folder with Parquet files
    npartitions : int
                  Number of Parquet files in which the resulting
                  coordinates will be written into.
    x_coords : xarray.DataArray
               Horizontal coordinates of pixel locations
    y_coords : xarray.DataArray
               Vertical coordinates of pixel locations
    chip_res : int
               Size of the chip in number of pixels
               
    Returns
    -------
    out_p : str
            Path to the folder with Parquet files (this can be passed
            to `ddf.from_parquet`)
    chip_len : float
               Length of the chip side expressed in the same units
               as `x_coords` and `y_coords`
    '''   
    if out_p[-1] != '/':
        out_p += '/'
    if not os.path.isdir(out_p):
        os.mkdir(out_p)
    chip_xs = x_coords.coarsen(
        {x_coords.name: chip_res}, boundary='trim'
    ).mean()
    chip_ys = y_coords.coarsen(
        {y_coords.name: chip_res}, boundary='trim'
    ).mean()
    chip_len = float(chip_xs[1] - chip_xs[0])
    # Pick largest dimension to split
    dims = {'x': x_coords, 'y': y_coords}
    dim_to_split = [
        i for i in dims if len(dims[i]) == max(
            [len(dims[j]) for j in dims]
        )
    ][0]
    dim_touse_full = [i for i in dims if i is not dim_to_split][0]
    # Split over largest dimension
    dts_n = dims[dim_to_split].shape[0]
    chunk_size = int(dts_n / npartitions)
    chunk_borders = [
        i * chunk_size for i in range(npartitions)
    ] + [dts_n]
    # Build bag
    items = []
    for i in range(npartitions):
        job_coords = {
            dim_to_split: dims[dim_to_split][
                chunk_borders[i]: chunk_borders[i+1]
            ],
            dim_touse_full: dims[dim_touse_full]
        }
        item = (
            out_p+f'chunk_{i}.pq',
            chunk_borders[i],
            job_coords['x'].values,
            job_coords['y'].values
        )
        items.append(item)
    # Execute computation
    _ = list(map(_build_xys_write, items))
    '''
    import dask.bag as dbag
    bag = dbag.from_sequence(items)
    out = bag.map(_build_xys_write)
    '''
    return out_p, chip_len


def _build_xys_write(pars):
    out_p, start, x_coords, y_coords = pars
    xys = cartesian((x_coords, y_coords))
    xys = pandas.DataFrame(
        xys, 
        index=pandas.RangeIndex(start=start, stop=start+xys.shape[0]),
        columns=['X', 'Y']
    )
    xys.to_parquet(out_p)
    return out_p


@njit(parallel=True)
def cartesian(arrays):
    '''
    Source: https://gist.github.com/hernamesbarbara/68d073f551565de02ac5#gistcomment-3527213
    '''
    n = 1
    for x in arrays:
        n *= x.size
    out = np.zeros((n, len(arrays)), dtype=arrays[0].dtype)

    for i in range(len(arrays)):
        m = int(n / arrays[i].size)
        out[:n, i] = np.repeat(arrays[i], m)
        n //= arrays[i].size

    n = arrays[-1].size
    for k in range(len(arrays)-2, -1, -1):
        n *= arrays[k].size
        m = int(n / arrays[k].size)
        for j in range(1, arrays[k].size):
            out[j*m:(j+1)*m,k+1:] = out[0:m,k+1:]
    return out




#-------------------------------------------------
#              DEPRECATED
#-------------------------------------------------
def coords2xys_old(x_coords, y_coords, chip_res):
    chip_xs = x_coords.coarsen(
        {x_coords.name: chip_res}, boundary='trim'
    ).mean()
    chip_ys = y_coords.coarsen(
        {y_coords.name: chip_res}, boundary='trim'
    ).mean()
    chip_len = float(chip_xs[1] - chip_xs[0])
    chip_xys = np.meshgrid(chip_xs.values, chip_ys.values)
    chip_xys = np.vstack(
        (chip_xys[0].flatten(), chip_xys[0].flatten())
    )
    return chip_xys, chip_len
