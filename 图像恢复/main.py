# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
import cv2  # opencv库
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # 回归分析


def read_image(img_path):
    """
    读取图片，图片是以 np.array 类型存储
    :param img_path: 图片的路径以及名称
    :return: img np.array 类型存储
    """
    # 读取图片
    img = cv2.imread(img_path)
    
    # 如果图片是三通道，采用 matplotlib 展示图像时需要先转换通道
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    return img


def plot_image(image, image_title, is_axis=False):
    """
    展示图像
    :param image: 展示的图像，一般是 np.array 类型
    :param image_title: 展示图像的名称
    :param is_axis: 是否需要关闭坐标轴，默认展示坐标轴
    :return:
    """
    # 展示图片
    plt.imshow(image)
    
    # 关闭坐标轴,默认关闭
    if not is_axis:
        plt.axis('off')

    # 展示受损图片的名称
    plt.title(image_title)

    # 展示图片
    plt.show()


def save_image(filename, image):
    """
    将np.ndarray 图像矩阵保存为一张 png 或 jpg 等格式的图片
    :param filename: 图片保存路径及图片名称和格式
    :param image: 图像矩阵，一般为np.array
    :return:
    """
    # np.copy() 函数创建一个副本。
    # 对副本数据进行修改，不会影响到原始数据，它们物理内存不在同一位置。
    img = np.copy(image)
    
    # 从给定数组的形状中删除一维的条目f
    img = img.squeeze()
    
    # 将图片数据存储类型改为 np.uint8
    if img.dtype == np.double:
        
        # 若img数据存储类型是 np.double ,则转化为 np.uint8 形式
        img = img * np.iinfo(np.uint8).max
        
        # 转换图片数组数据类型
        img = img.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 生成图片
    cv2.imwrite(filename, img)


def normalization(image):
    """
    将数据线性归一化
    :param image: 图片矩阵，一般是np.array 类型 
    :return: 将归一化后的数据，在（0,1）之间
    """
    # 获取图片数据类型对象的最大值和最小值
    info = np.iinfo(image.dtype)
    
    # 图像数组数据放缩在 0-1 之间
    return image.astype(np.double) / info.max


def noise_mask_image(img, noise_ratio):
    """
    根据题目要求生成受损图片
    :param img: 图像矩阵，一般为 np.ndarray
    :param noise_ratio: 噪声比率，可能值是0.4/0.6/0.8
    :return: noise_img 受损图片, 图像矩阵值 0-1 之间，数据类型为 np.array, 
             数据类型对象 (dtype): np.double, 图像形状:(height,width,channel),通道(channel) 顺序为RGB
    """
    # 受损图片初始化
    noise_img = None

    # -------------实现受损图像答题区域-----------------
    row, col = img.shape[0], img.shape[1]
    rgb = [None, None, None]  # rgb
    for i in range(3):
        # 构造其中一个通道的噪声图
        for j in range(row):
            if rgb[i] is None:
                rgb[i] = np.random.choice(2, (1, col), p=[noise_ratio, 1-noise_ratio])
            else:
                a = np.random.choice(2, (1, col), p=[noise_ratio, 1-noise_ratio])
                rgb[i] = np.concatenate((rgb[i], a), axis=0)

    # 扩展 shape
    for i in range(3):
        rgb[i] = rgb[i][:, :, np.newaxis]
    # 合并
    rst = np.concatenate((rgb[0], rgb[1], rgb[2]), axis=2)
    noise_img = rst * img
    # -----------------------------------------------

    return noise_img


def get_noise_mask(noise_img):
    """
    获取噪声图像，一般为 np.array
    :param noise_img: 带有噪声的图片
    :return: 噪声图像矩阵
    """
    # 将图片数据矩阵只包含 0和1,如果不能等于 0 则就是 1。
    return np.array(noise_img != 0, dtype='double')


def compute_error(res_img, img):
    """
    计算恢复图像 res_img 与原始图像 img 的 2-范数
    :param res_img:恢复图像 
    :param img:原始图像 
    :return: 恢复图像 res_img 与原始图像 img 的2-范数
    """
    # 初始化
    error = 0.0
    
    # 将图像矩阵转换成为np.narray
    res_img = np.array(res_img)
    img = np.array(img)
    
    # 如果2个图像的形状不一致，则打印出错误结果，返回值为 None
    if res_img.shape != img.shape:
        print("shape error res_img.shape and img.shape %s != %s" % (res_img.shape, img.shape))
        return None
    
    # 计算图像矩阵之间的评估误差
    error = np.sqrt(np.sum(np.power(res_img - img, 2)))
    
    return round(error,3)


def restore_image(noise_img, size=4):
    """
    使用 区域二元线性回归模型 进行图像恢复。
    :param noise_img: 一个受损的图像
    :param size: 输入区域半径，长宽是以 size*size 方形区域获取区域, 默认是 4
    :return: res_img 恢复后的图片，图像矩阵值 0-1 之间，数据类型为 np.array,
            数据类型对象 (dtype): np.double, 图像形状:(height,width,channel), 通道(channel) 顺序为RGB
    """
    # 恢复图片初始化，首先 copy 受损图片，然后预测噪声点的坐标后作为返回值。
    res_img = np.copy(noise_img)

    # 获取噪声图像
    noise_mask = get_noise_mask(noise_img)

    # -------------实现图像恢复代码答题区域----------------------------
    rows, cols, channel = res_img.shape
    region = 10  # 10 * 10
    row_cnt = rows // region
    col_cnt = cols // region

    for chan in range(channel):
        for rn in range(row_cnt + 1):
            ibase = rn * region
            if rn == row_cnt:
                ibase = rows - region
            for cn in range(col_cnt + 1):
                jbase = cn * region
                if cn == col_cnt:
                     jbase = cols - region
                x_train = []
                y_train = []
                x_test = []
                for i in range(ibase, ibase+region):
                    for j in range(jbase, jbase+region):
                        if noise_mask[i, j, chan] == 0:  # 噪音点
                            x_test.append([i, j])
                            continue
                        x_train.append([i, j])
                        y_train.append([res_img[i, j, chan]])
                if x_train == []:
                    print("x_train is None")
                    continue
                reg = LinearRegression()
                reg.fit(x_train, y_train)
                pred = reg.predict(x_test)
                for i in range(len(x_test)):
                    res_img[x_test[i][0], x_test[i][1], chan] = pred[i][0]
    res_img[res_img > 1.0] = 1.0
    res_img[res_img < 0.0] = 0.0
    # ---------------------------------------------------------------
    return res_img


if __name__ == "__main__":
    img_path = 'A.png'
    img = read_image(img_path)
    

    noise_ratio = 0.8
    nor_img = normalization(img)
    print(nor_img.shape)
    
    # 生成受损图片
    noise_img = noise_mask_image(nor_img, noise_ratio)

    # 恢复图片
    res_img = restore_image(noise_img)

    
    print("噪声图和原图之间的误差: {}".format(compute_error(noise_img, nor_img)))
    print("恢复图和原图之前的误差: {}".format(compute_error(res_img, nor_img)))


    # 展示恢复图片
    # plot_image(image=res_img, image_title="restore image")

    # 保存恢复图片
    save_image('res_' + img_path, nor_img)
    
            
