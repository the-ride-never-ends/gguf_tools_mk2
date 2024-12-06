
import numpy as np
import numpy.typing as npt
import rasterio

def write_array_to_geotiff(
        array: npt.NDArray, 
        output_path: str, 
        crs=None, # TODO Maybe there's a way to define a coordinate reference system for tensors???
        transform=None):
    """
    Write a numpy array to a GeoTIFF file using rasterio.

    Args:
        array (numpy.ndarray): The 2D or 3D numpy array to write.
        output_path (str): The path where the GeoTIFF file will be saved.
        crs (rasterio.crs.CRS, optional): The coordinate reference system. Default is None.
        transform (affine.Affine, optional): The affine transform. Default is None.

    Example usage:
        array = np.random.rand(256, 256).astype(np.float32)
        write_array_to_geotiff(array, 'output.tif')

    """
    # Get the number of bands and array dimensions
    if array.ndim == 2:
        height, width = array.shape
        bands = 1
    elif array.ndim == 3:
        bands, height, width = array.shape
    else:
        raise ValueError("Input array must be 2D or 3D")

    # Define the data type
    match array.dtype:
        case np.float32:
            dtype = rasterio.float32
        case np.float64:
            dtype = rasterio.float64
        case np.int16:
            dtype = rasterio.int16
        case np.int32:
            dtype = rasterio.int32
        case np.uint8:
            dtype = rasterio.uint8
        case _:
            raise ValueError(f"Unsupported data type: {array.dtype}")

    # Create a new GeoTIFF file
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dataset:
        # Write the array to the file
        if bands == 1:
            dataset.write(array, 1)
        else:
            for band in range(bands):
                dataset.write(array[band], band + 1)

    print(f"GeoTIFF file saved to {output_path}")
