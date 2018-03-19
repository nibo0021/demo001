#coding = utf-8
#tensorflow  构建数据集，数据集分为两类，每一类为一个文件夹，文件夹的名称为标签名
#tensorflow  读取数据集


import os
import tensorflow as tf
from PIL import Image

#获取数据集的路径
cwd = os.getcwd()                            # 获取当前工程的路径
trainset_path = "./data/train/"
testset_path  = "./data/test/"
classes = os.listdir(trainset_path)           # 获取当前路径中文件夹的数量,即类型的数量


def create_to_tfrecords(tfrecord_name,Flag="train",reszie=(50,50)):
    """

    :param tfrecord_name: tfrecords的名称
    :param Flag:     判断是“train” or “test”,默认是train
    :param reszie:
    :return:
    """
    #样本的数量
    num_example =0

    # 设置tfrecords文件的名字和路径
    writer = tf.python_io.TFRecordWriter(tfrecord_name)

    # 设置特征转换的函数
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    for index, name in enumerate(classes):

        if Flag == "train":
            class_path = trainset_path + name + "/"  # 图像文件的路径
        else:
            class_path = testset_path +name+"/"      #测试图片的路径


        # 输出文件的名字
        print(class_path)

        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name
                # 读取文件
                img = Image.open(img_path)
                # 设置图像文件的大小
                img = img.resize((reszie, reszie))
                # 将图像转化为原生二进制
                img_raw = img.tobytes()
                # 将数据整理成为TFRecord需要的数据结构

                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=_int64_feature(int[name])),
                    "img_raw": tf.train.Feature(bytes_list=_bytes_feature(img_raw))
                }))
                writer.write(example.SerializeToString())  # 序列化字符串
                print(img_name)
                num_example+=1
    print("样本的数量：", num_example)
    writer.close()


#读取tfrecord的二进制数据
def read_and_decode(filename,img_size):  # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [img_size, img_size, 3])  # reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
    label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
    return img, label





#根据队列刘数据格式，读取一批图片后，输入一批图片，对其做预处理，及样本的随机扩充
def get_batch(batch_size,crop_size):

    img,label = read_and_decode("train.tfrecords",50)
    #数据扩充变换
    distorted_image = tf.random_crop(img, [crop_size, crop_size, 3])                   # 随机裁剪
    distorted_image = tf.image.random_flip_up_down(distorted_image)                    # 上下随机翻转
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)         # 亮度变化
    distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)   # 对比度变化

    #生成batch
    #shuffle_batch的参数：capacity用于定义shuttle的范围，如果是对整个训练数据集，获取batch，那么capacity就应该够大
    #保证数据打的足够乱
    img_batch, label_batch = tf.train.shuffle_batch([distorted_image, label],batch_size=batch_size,
                                                 num_threads=16,capacity=50000,min_after_dequeue=10000)
    label_batch = tf.one_hot(label_batch,2,1,0)

    return  img_batch,label_batch




if __name__ == '__main__' :
    create_to_tfrecords("train.tfrecords")
    image_batch,label_batch = get_batch(100,20)
    print("数据的大小：", image_batch.shape)
    print("标签的大小:" , label_batch.shape)
    with tf.Session() as sess: #开始一个会话
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        for i in range(20):
            example, l = sess.run([image_batch,label_batch])#在会话中取出image和label
            img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
            img.save(cwd+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
            print(example, l)
        coord.request_stop()
        coord.join(threads)




