# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:46:52 2018

@author: Administrator
"""

import tensorflow as tf 
import os
from  PIL import  Image
import  cv2
import  numpy as np

image_size = 128

tf_save_path="F:\\DeepLearning_Codes\\study\\Demo001\\data\\my_tfrecords\\train.tfrecords"
#样本分类的标签文件
label_path = "F:\\DeepLearning_Codes\\study\\Demo001\\data\\labels.txt"
#样本的原图的路径
image_path = "F:\\DeepLearning_Codes\\study\\Demo001\\data\\trainImage\\"
#反向验证的图片输出的路径
tf_out_path = "F:\\DeepLearning_Codes\\study\\Demo001\\data\\tf_out\\ "

f=open(label_path)
class_id_cnd = 0 #类型的个数
classes_read=[]

print ("\n读取样本的数量")
while True:
    line = f.readline()
    if line:
        class_id_cnd = class_id_cnd + 1
        line = line.strip()
        classes_read.append(line)
        print(class_id_cnd,")","--",classes_read[class_id_cnd-1])
    else:
        break
f.close()
print("\n")

writer = tf.python_io.TFRecordWriter(tf_save_path)
picture_cnt = 0  #记录样本的总数量

for index,name in enumerate(classes_read):
    class_path = image_path+name+'\\'""
    print("第",index,"类开始转换~~~")
    for img_name in os.listdir(class_path):
        img_path = class_path+img_name
        #opencv 显示
        # if(picture_cnt%10 ==0):
        #     img_cv = cv2.imread(img_path)
        #     cv2.namedWindow("image_tfrecords")
        #     cv2.imshow("image_tfrecords",img_cv)
        #     cv2.waitKey(1)
        #将样本图像写入文件train.tfrecords
        img  = Image.open(img_path)
        img = img.resize((image_size,image_size))
        img_raw = img.tobytes()   #将28 * 28 图像转化为二进制
        #example对象样本标签和图像数据进行封装
        example=tf.train.Example(features=tf.train.Features(feature={
            'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw':tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
        picture_cnt=picture_cnt+1
writer.close()
cv2.destroyAllWindows()

print("Tfrecords文件生成：",picture_cnt)

#fuction:读取tfrecords文件功能函数
def read_and_derecord(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader= tf.TFRecordReader()
    _,serialized_example=reader.read(filename_queue)
    features= tf.parse_single_example(serialized_example,features={
        'label':tf.FixedLenFeature([],tf.int64),
        'img_raw':tf.FixedLenFeature([],tf.string)
    })
    img = tf.decode_raw(features['img_raw'],tf.uint8)
    img = tf.reshape(img, [image_size, image_size, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label=tf.cast(features['label'],tf.int64)
    return  img,label

image2, label2 = read_and_derecord(tf_save_path)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    #启动多线程
    threads=tf.train.start_queue_runners(coord=coord)
    Classes_Cnt=np.zeros([class_id_cnd],np.int32)   #记录每个分类的个数
    for i in range(picture_cnt):
        example,class_num=sess.run([image2,label2])
        if(i%30==0):
            cv2.namedWindow("image_out",0)
            cv2.imshow("image_out",example)
            cv2.waitKey(1)
        out_file=tf_out_path+str(i)+"_lable_"+str(class_num)+".jpg"
        cv2.imwrite(out_file,example) #存储灰度图像
        Classes_Cnt[class_num]=Classes_Cnt[class_num]+1
    coord.request_stop()
    coord.join(threads)

    #打印每个类型的样本个数
    for i in range(class_id_cnd):
        print("分类号",i,"=",Classes_Cnt[i],"个样本数")
    cv2.destroyAllWindows() #销毁所有的窗口
    sess.close()
    print("\n tfrecords测试转换成功")
    print("well done")




