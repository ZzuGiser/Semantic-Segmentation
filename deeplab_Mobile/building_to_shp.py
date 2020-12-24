#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# !/usr/bin/env python
# -*- coding:utf-8 -*-
from PIL import Image

import datetime
import logging
import geopandas as gpd
import sys
from nets.deeplab import Deeplabv3
from PIL import Image
import numpy as np
import copy
import os
from matplotlib.patches import Polygon

from skimage.measure import find_contours

try:
    from osgeo import gdal
    from osgeo import ogr
    from osgeo import osr
except ImportError:
    import gdal
    import ogr
    import osr

# Root directory of the project
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Download COCO trained weights from Releases if needed
TIF_PATH = "../../tif_and_shp/CJ2.tif"
SHP_PATH = "../../tif_and_shp/shp_building/guangzhou.shp"
CROP_SIZE = 200
NEW_SHP_NAME = 'building_detect.shp'
ALL_NUM = 10000
CLASS_NAMES = ['not_defined', 'building']
OUTPUT_PATH = os.path.join(ROOT_DIR, 'result')

class_colors = [[0, 0, 0], [0, 255, 0]]
NCLASSES = 2
HEIGHT = 416
WIDTH = 416
MODEL = Deeplabv3(classes=2, input_shape=(HEIGHT, WIDTH, 3))
MODEL.load_weights("logs/last1.h5")



class TIF_TRANS(object):
    def __init__(self, path=TIF_PATH):
        gdal.AllRegister()
        self.dataset = gdal.Open(path)

    def imagexy2geo(self, row, col):
        '''
            根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
            :param dataset: GDAL地理数据
            :param row: 像素的行号
            :param col: 像素的列号
            :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
        '''
        trans = self.dataset.GetGeoTransform()
        px = trans[0] + col * trans[1] + row * trans[2]
        py = trans[3] + col * trans[4] + row * trans[5]
        return px, py

    def geo2imagexy(self, x, y):
        '''
        根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
        :param dataset: GDAL地理数据
        :param x: 投影或地理坐标x
        :param y: 投影或地理坐标y
        :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
        '''
        trans = self.dataset.GetGeoTransform()
        a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
        b = np.array([x - trans[0], y - trans[3]])
        return np.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解


class Remote2Shp(object):
    def __init__(self, tif_path=TIF_PATH, shp_path=SHP_PATH, new_shp_path=NEW_SHP_NAME, model=MODEL,
                 class_names=CLASS_NAMES,
                 output_path=OUTPUT_PATH):
        self.model = model
        self.class_names = class_names
        self.ouput_path = output_path
        self.tif_path = tif_path
        self.tif_img = gdal.Open(tif_path)
        self.shp_data = gpd.read_file(shp_path)
        self.image_num = 0
        self.shp_img = np.ones((self.tif_img.RasterXSize, self.tif_img.RasterYSize)) * 255
        self.new_shp_path = os.path.join(output_path, new_shp_path)
        self.oDS = self.init_create_shp()

    def init_create_shp(self):
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")  # 为了支持中文路径
        gdal.SetConfigOption("SHAPE_ENCODING", "CP936")  # 为了使属性表字段支持中文
        strVectorFile = self.new_shp_path  # 定义写入路径及文件名
        ogr.RegisterAll()  # 注册所有的驱动
        strDriverName = "ESRI Shapefile"  # 创建数据，这里创建ESRI的shp文件
        oDriver = ogr.GetDriverByName(strDriverName)
        if oDriver == None:
            print("%s 驱动不可用！\n", strDriverName)
        oDS = oDriver.CreateDataSource(strVectorFile)  # 创建数据源
        if oDS == None:
            print("创建文件【%s】失败！", strVectorFile)
        return oDS

    def creaate_val_sample(self, crop_size=CROP_SIZE):
        srs = osr.SpatialReference()  # 创建空间参考
        srs.ImportFromEPSG(4326)  # 定义地理坐标系WGS1984
        papszLCO = []
        # 创建图层，创建一个多边形图层,"TestPolygon"->属性表名
        oLayer = self.oDS.CreateLayer("TestPolygon", srs, ogr.wkbPolygon, papszLCO)
        if oLayer == None:
            print("图层创建失败！\n")
        '''下面添加矢量数据，属性表数据、矢量数据坐标'''
        oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)  # 创建一个叫FieldID的整型属性
        oLayer.CreateField(oFieldID, 1)
        oFieldName = ogr.FieldDefn("FieldName", ogr.OFTString)  # 创建一个叫FieldName的字符型属性
        oFieldName.SetWidth(100)  # 定义字符长度为100
        oLayer.CreateField(oFieldName, 1)
        tif_tran = TIF_TRANS(self.tif_path)
        for shp_i, geo in enumerate(self.shp_data.geometry):
            if shp_i > ALL_NUM:
                break
            row, col = tif_tran.geo2imagexy(geo.centroid.x, geo.centroid.y)
            x_df, y_df = int(crop_size / 2), int(crop_size / 2)
            raster_crop = self.tif_crop(crop_size, row, col, x_df, y_df)
            if len(raster_crop) == 0:
                continue
            image = raster_crop
            if len(image.shape) == 2:
                image = image[:, :, np.newaxis]
                image = np.concatenate((image, image, image), axis=2)
            else:
                image = np.stack(image, axis=2)
            w, h, _ = image.shape  # w = 400,h = 400
            self.handle_img(image, oLayer,[row, col], tif_tran,shp_i)

            # results = model.detect([image], verbose=1)
            # # Visualize results
            # r = results[0]
            # visualize.add_instances(r['rois'], r['masks'], r['class_ids'], oLayer, [row, col], tif_tran,shp_i)
            # img_out_path = os.path.join(self.ouput_path,"{}.jpg".format(str(shp_i)))
            # image = image.astype(np.uint8)
            # image = Image.fromarray(image).convert('RGB')
            # image.save(img_out_path)
        self.oDS.Destroy()
        print("数据集创建完成！\n")

    def handle_img(self, img, oLayer,origin_point, tif_tran,shp_i):
        pr = self.model.predict(img)[0]
        pr = pr.reshape((int(HEIGHT), int(WIDTH), NCLASSES)).argmax(axis=-1)
        # seg_img = np.zeros((int(HEIGHT), int(WIDTH), 3))
        # colors = class_colors
        # for c in range(NCLASSES):
        #     seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
        #     seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
        #     seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
        # seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
        contours = find_contours(pr, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            oDefn = oLayer.GetLayerDefn()  # 定义要素
            # 创建单个面
            oFeatureTriangle = ogr.Feature(oDefn)
            oFeatureTriangle.SetField(0, shp_i)  # 第一个参数表示第几个字段，第二个参数表示字段的值
            oFeatureTriangle.SetField(1, 'building')
            ring = ogr.Geometry(ogr.wkbLinearRing)  # 构建几何类型:线
            for point in verts:
                point = tif_tran.imagexy2geo(origin_point[1] + point[1] - 200, origin_point[0] + point[0] - 200)
                ring.AddPoint(float(point[0]), float(point[1]))
            yard = ogr.Geometry(ogr.wkbPolygon)  # 构建几何类型:多边形
            yard.AddGeometry(ring)
            yard.CloseRings()
            geomTriangle = ogr.CreateGeometryFromWkt(str(yard))  # 将封闭后的多边形集添加到属性表
            oFeatureTriangle.SetGeometry(geomTriangle)
            oLayer.CreateFeature(oFeatureTriangle)



    def tif_crop(self, crop_size, x, y, x_df, y_df):
        dataset_img = self.tif_img
        width = dataset_img.RasterXSize
        height = dataset_img.RasterYSize
        img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据

        #  获取当前文件夹的文件个数len,并以len+1命名即将裁剪得到的图像
        new_name = '{}_{}_{}.jpg'.format(self.image_num, int(x), int(y))
        #  裁剪图片,重复率为RepetitionRate
        x_min, x_max = x - x_df, x + crop_size - x_df
        y_min, y_max = y - y_df, y + crop_size - y_df

        if (len(img.shape) == 2):
            cropped = img[int(y_min): int(y_max), int(x_min): int(x_max)]
        # 如果图像是多波段
        else:
            if img.shape[0] > 3:
                cropped = img[0:3, int(y_min): int(y_max),
                          int(x_min): int(x_max)]
            else:
                cropped = img[:, int(y_min): int(y_max),
                          int(x_min): int(x_max)]
        # 写图像
        if x_min < 0 or x_max > height or y_min < 0 or y_max > width:
            return []

        self.image_num += 1
        logging.info('crop image name:{}'.format(new_name))
        return cropped

    def remote2Shp(self):
        self.creaate_val_sample()


if __name__ == '__main__':

    model = MODEL
    class_names = CLASS_NAMES
    output_pack = '{:%Y%m%d_%H%M}_building_to_shp'.format(datetime.datetime.now())
    output_path = os.path.join(OUTPUT_PATH, output_pack)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(output_path, 'a_reslut.log'),
                        filemode='w')
    remote2shp = Remote2Shp(output_path=output_path)
    remote2shp.remote2Shp()
