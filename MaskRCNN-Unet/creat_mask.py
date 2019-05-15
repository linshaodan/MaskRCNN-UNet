from osgeo import gdal, ogr
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


def create_building_mask(rasterSrc, vectorSrc, npDistFileName='',
                         noDataValue=0, burn_values=1):
    '''
    Create building mask for rasterSrc,
    Similar to labeltools/createNPPixArray() in spacenet utilities
    '''

    ## open source vector file that truth data
    source_ds = ogr.Open(vectorSrc)
    source_layer = source_ds.GetLayer()

    ## extract data from src Raster File to be emulated
    ## open raster file that is to be emulated
    srcRas_ds = gdal.Open(rasterSrc)
    cols = srcRas_ds.RasterXSize
    rows = srcRas_ds.RasterYSize

    ## create First raster memory layer, units are pixels
    # Change output to geotiff instead of memory
    memdrv = gdal.GetDriverByName('GTiff')
    dst_ds = memdrv.Create(npDistFileName, cols, rows, 1, gdal.GDT_Byte,
                           options=['COMPRESS=LZW'])
    dst_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    dst_ds.SetProjection(srcRas_ds.GetProjection())
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(noDataValue)
    gdal.RasterizeLayer(dst_ds, [1], source_layer, burn_values=[burn_values])
    dst_ds = 0

    return
def create_poly_mask(rasterSrc, vectorSrc, npDistFileName='',
                            noDataValue=0, burn_values=1):

    '''
    Create polygon mask for rasterSrc,
    Similar to labeltools/createNPPixArray() in spacenet utilities
    '''

    ## open source vector file that truth data
    source_ds = ogr.Open(vectorSrc)
    source_layer = source_ds.GetLayer()

    ## extract data from src Raster File to be emulated
    ## open raster file that is to be emulated
    srcRas_ds = gdal.Open(rasterSrc)
    cols = srcRas_ds.RasterXSize
    rows = srcRas_ds.RasterYSize

    if npDistFileName == '':
        dstPath = ".tmp.tiff"
    else:
        dstPath = npDistFileName

    ## create First raster memory layer, units are pixels
    # Change output to geotiff instead of memory
    memdrv = gdal.GetDriverByName('GTiff')
    dst_ds = memdrv.Create(dstPath, cols, rows, 1, gdal.GDT_Byte,
                           options=['COMPRESS=LZW'])
    dst_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    dst_ds.SetProjection(srcRas_ds.GetProjection())
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(noDataValue)
    gdal.RasterizeLayer(dst_ds, [1], source_layer, burn_values=[burn_values])
    dst_ds = 0

    mask_image = Image.open(dstPath)
    mask_image = np.array(mask_image)

    if npDistFileName == '':
        os.remove(dstPath)

    return mask_image

image_id = 1062
# img_dir = "/home/ly/data/dl_data/spacenet/AOI_5_Khartoum_Train/RGB-PanSharpen"
# json_dir = "/home/ly/data/dl_data/spacenet/AOI_5_Khartoum_Train/geojson/buildings"
# img_name = "RGB-PanSharpen_AOI_5_Khartoum_img{}.tif".format(image_id)
# json_name = "buildings_AOI_5_Khartoum_img{}.geojson".format(image_id)
# img_path = os.path.join('F:\遥感数据\\road\SpaceNet', img_name)
# json_path = os.path.join(json_dir, json_name)

json_path = "F:\遥感数据\\road\SpaceNet\\buildings_AOI_5_Khartoum_img984.geojson"
img_path = "F:\遥感数据\\road\SpaceNet\AOI_5_Khartoum_Train\RGB-PanSharpen\\RGB-PanSharpen_AOI_5_Khartoum_img984.tif"
create_building_mask(img_path, json_path,"F:\遥感数据\\road\SpaceNet\\mask984.tif")
mask = plt.imread("F:\遥感数据\\road\SpaceNet\\mask984.tif")
plt.imshow(mask)
plt.show( )


"""
def create_building_mask(rasterSrc, vectorSrc, npDistFileName='', 
                            noDataValue=0, burn_values=1):

    '''
    Create building mask for rasterSrc,
    Similar to labeltools/createNPPixArray() in spacenet utilities
    '''
    
    ## open source vector file that truth data
    source_ds = ogr.Open(vectorSrc)
    source_layer = source_ds.GetLayer()

    ## extract data from src Raster File to be emulated
    ## open raster file that is to be emulated
    srcRas_ds = gdal.Open(rasterSrc)
    cols = srcRas_ds.RasterXSize
    rows = srcRas_ds.RasterYSize

    ## create First raster memory layer, units are pixels
    # Change output to geotiff instead of memory 
    memdrv = gdal.GetDriverByName('GTiff') 
    dst_ds = memdrv.Create(npDistFileName, cols, rows, 1, gdal.GDT_Byte, 
                           options=['COMPRESS=LZW'])
    dst_ds.SetGeoTransform(srcRas_ds.GetGeoTransform())
    dst_ds.SetProjection(srcRas_ds.GetProjection())
    band = dst_ds.GetRasterBand(1)
    band.SetNoDataValue(noDataValue)    
    gdal.RasterizeLayer(dst_ds, [1], source_layer, burn_values=[burn_values])
    dst_ds = 0
    
    return
"""