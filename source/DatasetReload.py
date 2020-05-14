# -*- coding: utf-8 -*-
"""
Created on 2019.12.4
latest modify 2020.5.10
@author: Junbin
@note: DatasetReload
"""

import numpy as np
import os
import io
import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import warnings


class PointCloud():
    def __init__(self,IsTrain=True,CrossValidation_pro=0.3,num_classes=None,input_shape=None,dataset_dir=None):
        self.IsTrain               = IsTrain    #训练or测试
        self.CrossValidation_pro   = CrossValidation_pro  #交叉验证百分比
        if num_classes is None:
            self.num_classes       = 2       #分类数目
        else:
            self.num_classes       = num_classes
        if input_shape is None:
            self.input_width       = 224     #输入的形状大小
            self.input_height      = 224
        else:
            self.input_width       = input_shape[input_width]
            self.input_height      = input_shape[input_height]
        if dataset_dir is None:
            self.dataset_dir       = './Datasets/'     #数据集存放的目录
        else:
            self.dataset_dir       = dataset_dir

    def adjust_dataset(self,train_data, train_label, test_data, test_label):
        '''
        @ note: 打乱数据集
        @ para: "数据集"
        @ para: "打乱后的数据集"
        '''
        train_num = len(train_label)
        test_num = len(test_label)
        train_seq_distruption = [i for i in range(0, train_num)]
        test_seq_distruption = [i for i in range(0, test_num)]
        random.shuffle(train_seq_distruption)
        random.shuffle(test_seq_distruption)
        distrupted_train_data = []
        distrupted_train_label = []
        distrupted_test_data = []
        distrupted_test_label = []
        for i in train_seq_distruption:
            distrupted_train_data.append(train_data[i])
            distrupted_train_label.append(train_label[i])
        for i in test_seq_distruption:
            distrupted_test_data.append(test_data[i])
            distrupted_test_label.append(test_label[i])
        return np.array(distrupted_train_data), np.array(distrupted_train_label), np.array(distrupted_test_data), np.array(distrupted_test_label)


    def DataTransform(self,filename,IsShowImage=False):
        '''
        @ note：通过pandas读取csv文件，转化为二维视图;
                当前置零x维度，暂不支持yz（当前采集数据大致垂直x平面）;
                并完成reshape，以符合网络的输入形状；
        @ para:
            filename 读取的csv文件名
        '''
        data = pd.read_csv(filename)            # 使用pandas库读取csv文件内容
        data.replace(np.nan, 0, inplace=True)   # 将数据为nan的地方替换为0
        data_y = ((data.y)*1000).astype('int64').values 
        data_z = ((data.z)*1000).astype('int64').values  
        reflectivity = (data.reflectivity).astype('uint8').values

        fig = plt.figure(dpi=200,figsize=(5,3),frameon=False)          # 设置窗口大小，像素
        canvas = fig.canvas
        plt.scatter(data_y,data_z,s=1,c=reflectivity,alpha=0.5)  # 绘制散点函数，具体参数意义查阅手册

        warnings.filterwarnings('ignore')
        plt.axes().get_xaxis().set_visible(False)   # 隐藏坐标轴
        plt.axes().get_yaxis().set_visible(False)   # 隐藏坐标轴

        buffer = io.BytesIO()     # 获取输入输出流对象
        canvas.print_png(buffer)  # 将画布上的内容打印到输入输出流对象
        iodata = buffer.getvalue()  # 获取流的值
        buffer.write(iodata)  # 将数据写入buffer
        image = Image.open(buffer)
        image_data = image.convert('RGB')
        image_data = image_data.resize((self.input_width,self.input_height))
        image_out = cv2.cvtColor(np.asarray(image_data),cv2.COLOR_RGB2BGR)  
        if IsShowImage is True:
            image_data.show()
        return image_out


    def __call__(self):
        '''
        @ note: 数据集读取
        @ para: None
        @ return: 
            IsTrain(True) :TrainDataset,TrainLable,ValDataset,ValLable
                   (False):TestDataset,TestLable
        '''
        TrainDataset  = []
        TrainLable    = []
        ValDataset    = []
        ValLable      = []
        TestDataset   = []
        TestLable     = []

        print("数据预处理进行中……\n")
        for i in range(0, self.num_classes):
            if self.IsTrain is True:
                data_dir = self.dataset_dir+'Train/'+str(i)+'/'
                assert os.path.exists(data_dir),"\n**Loading Datasets error,please check**\n"
            else:
                data_dir = self.dataset_dir+'Test/'+str(i)+'/'
                assert os.path.exists(data_dir),"\n**Loading Datasets error,please check**\n"
            datafiles = os.listdir(data_dir)
            DataNum  = len(datafiles)
            
            if self.IsTrain is True:
                CrossValidation_DataNum = int(self.CrossValidation_pro*DataNum)
                Train_DataNum = DataNum-CrossValidation_DataNum
                Dataset_sub=[]
                for j in range(0,DataNum):
                    Dataset_sub.append(j)
                Train_sub = sorted(random.sample(Dataset_sub,Train_DataNum))
                for k in range(0,DataNum):
                    assert os.path.exists(data_dir+datafiles[k]),"\n**Loading Datasets error,please check**\n"
                    image_data = self.DataTransform(data_dir+datafiles[k],False)
                    image_data = image_data / 255.
                    if k in Train_sub:
                        TrainDataset.append(image_data)
                        lable = np.zeros(self.num_classes)
                        lable[i] = 1.0
                        lable = lable.tolist()
                        TrainLable.append(lable)
                    else:
                        ValDataset.append(image_data)
                        lable = np.zeros(self.num_classes)
                        lable[i] = 1.0
                        lable = lable.tolist()
                        ValLable.append(lable)
                print("第",i+1,"类，训练集数量:",Train_DataNum,"交叉验证集数量:",CrossValidation_DataNum)
            else:
                for j in range(0,DataNum):
                    assert os.path.exists(data_dir+datafiles[j]),"\n**Loading Datasets error,please check**\n" 
                    image_data = self.DataTransform(data_dir+datafiles[k],False)
                    image_data = image_data / 255.
                    TestDataset.append(image_data)
                    lable = np.zeros(self.num_classes)
                    lable[i] = 1.0
                    lable = lable.tolist()
                    TestLable.append(lable)
        print("数据预处理完毕")
        if self.IsTrain is True:
            return self.adjust_dataset(TrainDataset,TrainLable,ValDataset,ValLable)
        else:
            return TestDataset,TestLable

class dir_apply():
    def __init__(self,checkpoint_path='./checkpoint',logs_path='./logs'):
        self.checkpoint_path = checkpoint_path
        self.logs_path = logs_path
    def mkdir(self,name):
        '''创建文件夹'''
        isExists=os.path.exists(name)
        if not isExists:
            os.makedirs(name)
        return 0
    def __call__(self):
        self.mkdir(self.checkpoint_path)
        self.mkdir(self.logs_path)


                        

def main():
    warnings.filterwarnings('ignore')
    pointcloud = PointCloud()
    pointcloud()


if __name__ == '__main__':
    main()
