import os
import xml.etree.ElementTree as ET

VOC2007_path = r'D:\dataset\VOCdevkit\VOC2007'
dataset_path = r'D:\dataset\VOCdevkit\VOC2007\Annotations'
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']


def corner_to_center(wh, point):
    """
    input:
        wh = img_size,point = (xmin, ymin ,xmax ,ymax);
    return:
        归一化后的obj中心以及宽高(x,y,w,h);
    """
    obj_w = (point[2] - point[0]) / wh[0]
    obj_h = (point[3] - point[1]) / wh[1]
    obj_center = (point[0]/wh[0] + obj_w / 2, point[1] / wh[1] + obj_h / 2)
    return (obj_center[0], obj_center[1], obj_w, obj_h)


def convert_annotation_to_txt(image_id,out_file):
    """把图像image_id的xml标注文件转换为目标检测需要的label文件（txt）
        其中包含有多行，每行表示一个object，包含类别信息，以及bbox的中心坐标以及
        宽和高，并将四个物理量归一化
        """
    infile = open(dataset_path + "\%s" % image_id)
    image_id = image_id.split('.')[0] + ".jpg"

    tree = ET.parse(infile)
    root = tree.getroot()
    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    out_file.write(image_id + " ")
    Point = []
    for obj in root.iter("object"):
        difficult = obj.find("difficult").text
        cls = obj.find("name").text
        if cls not in CLASSES or difficult == 1:
            continue
        cls_id = CLASSES.index(cls)
        xmlbox = obj.find('bndbox')
        point = (int(xmlbox.find("xmin").text),
                 int(xmlbox.find("ymin").text),
                 int(xmlbox.find("xmax").text),
                 int(xmlbox.find("ymax").text),
                 )

        out_file.write(" ".join([str(a) for a in point])+ " " + str(cls_id)+ " ")
    out_file.write("\n")



def make_label_txt():
    """在 labels文件夹下创建image_id.txt，对应每个image_id.xml提取出的bbox信息"""
    filenames = os.listdir(dataset_path)
    out_file = open(VOC2007_path + "\label2"+'\%s.txt' % "test1", 'w')
    for filename in filenames:
        convert_annotation_to_txt(filename,out_file)


make_label_txt()









