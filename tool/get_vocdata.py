from torch.utils.data import Dataset
import os
import cv2
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
import numpy as np
import random
import torch

img_path = r'D:\dataset\VOCdevkit\VOC2007\JPEGImages'
img_annotations =r"D:\dataset\VOCdevkit\VOC2007\Annotations"
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']



def resize_image_with_coords(img, coords, input_size):
    h, w = img.shape[:2]
    scale = min(input_size/h, input_size/w)
    nh = int(h * scale)
    nw = int(w * scale)
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    dw = (input_size - nw) / 2
    dh = (input_size - nh) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    new_image = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for coord in coords:
        coord[0] = int((coord[0] * scale) + dw)
        coord[2] = int((coord[2] * scale) + dw)
        coord[1] = int((coord[1] * scale) + dh)
        coord[3] = int((coord[3] * scale) + dh)
    return new_image, coords







class VocDataset(Dataset):
    def __init__(self, img_path, img_annotations, CLASSES, is_train=True, class_num=20,
                 label_smooth_value=0.05, input_size=448, grid_size=64):
        self.label_smooth_value = label_smooth_value
        self.img_path = img_path
        self.img_annotations = img_annotations
        self.CLASSES = CLASSES
        self.input_size = input_size
        self.grid_size = grid_size
        self.is_train = is_train
        if self.is_train:
            self.img_name = []
            file = open(r"D:\dataset\VOCdevkit\VOC2007\ImageSets\Main\trainval.txt")
            file_data = file.readlines()
            for row in file_data:
                row = row.replace('\n', '.jpg')
                self.img_name.append(row)
        else:
            self.img_name = []
            file = open(r"D:\dataset\VOCdevkit\VOC2007\ImageSets\Main\test.txt")
            file_data = file.readlines()
            for row in file_data:
                row = row.replace('\n', '.jpg')
                self.img_name.append(row)

        # self.img_name = os.listdir(self.img_path)
        # print(self.img_name)

        #coco训练集mean = [0.471, 0.448, 0.408]
        #std = [0.234, 0.239, 0.242]
        #imagenet数据集的均值和方差（三分量顺序是RGB）
        #mean = [0.485, 0.456, 0.406]
        #std = [0.229, 0.224, 0.225]
        self.class_num = class_num
        self.transform_common = transforms.Compose([
            transforms.ToTensor(),  # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229))  # 归一化后.不容易产生梯度爆炸的问题
        ])

    def __getitem__(self, item):
        img_path = os.path.join(self.img_path, self.img_name[item])
        annotations_path = os.path.join(self.img_annotations, self.img_name[item].replace('.jpg', '.xml'))
        img = cv2.imread(img_path)
        tree = ET.parse(annotations_path)
        root = tree.getroot()
        coords = []
        for obj in root.iter("object"):
            cls = obj.find('name').text
            if cls not in self.CLASSES:
                continue
            bndbox = obj.find("bndbox")
            xmin = int(bndbox.find("xmin").text)
            xmax = int(bndbox.find("xmax").text)
            ymin = int(bndbox.find("ymin").text)
            ymax = int(bndbox.find('ymax').text)
            class_id = self.CLASSES.index(cls)
            coords.append([xmin, ymin, xmax, ymax, class_id])
        # 按目标面积大小排序
        coords.sort(key=lambda coord : (coord[2] - coord[0]) * (coord[3] - coord[1]))
        coords = torch.tensor(coords)

        # 训练
        if self.is_train:
            img, coords = self.random_flip(img, coords)
            img, coords = self.randomScale(img, coords)
            img = self.randomBlur(img)
            img = self.RandomBrightness(img)
            img = self.RandomHue(img)
            img = self.RandomSaturation(img)
            img, coords = self.randomCrop(img, coords)
            img, coords = self.randomShift(img, coords)
        img, coords = resize_image_with_coords(img, coords, self.input_size)
        for i in coords:
            cv2.rectangle(img,(i[0],i[1]),(i[2],i[3]),((255,0,0)))
            cv2.imshow('s',img)
            cv2.waitKey(0)

        img = self.transform_common(img)
        ground_truth = self.getGroundTruth(coords)
        return img, ground_truth

    def __len__(self):
        return len(self.img_name)

    def One_Hot(self, class_id):
        class_ont_hot = np.zeros([self.class_num])

        class_ont_hot[int(class_id.item())] = 1.0
        return class_ont_hot
    # 水平翻转

    def random_flip(self, im, boxes):
        if random.random() < 0.5:
            im_lr = np.fliplr(im).copy()
            h, w, _ = im.shape
            for i in range(len(boxes)):
                xmin = w - boxes[i][2]
                xmax = w - boxes[i][0]
                boxes[i][0] = xmin
                boxes[i][2] = xmax
            return im_lr, boxes
        return im, boxes
    # 固定住高度，以0.8-1.2伸缩宽度，做图像形变

    def randomScale(self, bgr, boxes):
        # 固定住高度，以0.8-1.2伸缩宽度，做图像形变
        if random.random() < 0.5:
            scale = random.uniform(0.8, 1.2)
            height, width, c = bgr.shape
            bgr = cv2.resize(bgr, (int(width * scale), height))
            for i in range(len(boxes)):
                xmin = scale * boxes[i][0]
                xmax = scale * boxes[i][2]
                boxes[i][0] = xmin
                boxes[i][2] = xmax
            return bgr, boxes
        return bgr, boxes
    # 模糊

    def randomBlur(self, bgr):
        if random.random() < 0.5:
            bgr = cv2.blur(bgr, (5, 5))
        return bgr

    # 调节亮度
    def RandomBrightness(self,bgr):
        if random.random() < 0.5:
            hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            v = v*adjust
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        return bgr
    # 调节色相

    def RandomHue(self, bgr):
        if random.random() < 0.5:
            hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            adjust = random.choice([0.5, 1.5])
            h = h * adjust
            h = np.clip(h, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h, s, v))
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        return bgr
    # 调节饱和度

    def RandomSaturation(self,bgr):
        if random.random() < 0.5:
            hsv =cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
            h,s,v = cv2.split(hsv)
            adjust = random.choice([0.5,1.5])
            s = s*adjust
            s = np.clip(s, 0, 255).astype(hsv.dtype)
            hsv = cv2.merge((h,s,v))
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        return bgr
    # 随机剪裁

    def randomCrop(self,bgr,boxes):
        if random.random() < 0.5:
            center = (boxes[:,2:4]+boxes[: , :2])/2
            height,width,c = bgr.shape
            h = random.uniform(0.6*height,height)
            w = random.uniform(0.6*width,width)
            x = random.uniform(0,width-w)
            y = random.uniform(0,height-h)
            x,y,h,w = int(x),int(y),int(h),int(w)

            center = center - torch.FloatTensor([[x,y]]).expand_as(center)
            mask1 = (center[:,0]>0) & (center[:,0]<w)
            mask2 = (center[:,1]>0) & (center[:,1]<h)
            mask = (mask1 & mask2).view(-1,1)

            boxes_in = boxes[mask.expand_as(boxes)].view(-1,5)
            if(len(boxes_in)==0):
                return bgr,boxes
            box_shift = torch.FloatTensor([[x,y,x,y, 0]]).expand_as(boxes_in)

            boxes_in = boxes_in - box_shift
            boxes_in[:,0]=boxes_in[:,0].clamp_(min=0,max=w)
            boxes_in[:,2]=boxes_in[:,2].clamp_(min=0,max=w)
            boxes_in[:,1]=boxes_in[:,1].clamp_(min=0,max=h)
            boxes_in[:,3]=boxes_in[:,3].clamp_(min=0,max=h)

            img_croped = bgr[y:y+h,x:x+w,:]
            return img_croped,boxes_in
        return bgr,boxes

    # 平移变换

    def randomShift(self,bgr,boxes):

        center = (boxes[:,2:4]+boxes[:,:2])/2
        if random.random() <0.5:
            height,width,c = bgr.shape
            after_shfit_image = np.zeros((height,width,c),dtype=bgr.dtype)
            after_shfit_image[:,:,:] = (104,117,123) #bgr
            shift_x = random.uniform(-width*0.2,width*0.2)
            shift_y = random.uniform(-height*0.2,height*0.2)
            #print(bgr.shape,shift_x,shift_y)
            #原图像的平移
            if shift_x>=0 and shift_y>=0:
                after_shfit_image[int(shift_y):,int(shift_x):,:] = bgr[:height-int(shift_y),:width-int(shift_x),:]
            elif shift_x>=0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),int(shift_x):,:] = bgr[-int(shift_y):,:width-int(shift_x),:]
            elif shift_x <0 and shift_y >=0:
                after_shfit_image[int(shift_y):,:width+int(shift_x),:] = bgr[:height-int(shift_y),-int(shift_x):,:]
            elif shift_x<0 and shift_y<0:
                after_shfit_image[:height+int(shift_y),:width+int(shift_x),:] = bgr[-int(shift_y):,-int(shift_x):,:]

            shift_xy = torch.FloatTensor([[int(shift_x),int(shift_y)]]).expand_as(center)
            center = center + shift_xy
            mask1 = (center[:,0] >0) & (center[:,0] < width)
            mask2 = (center[:,1] >0) & (center[:,1] < height)
            mask = (mask1 & mask2).view(-1,1)
            boxes_in = boxes[mask.expand_as(boxes)].view(-1,5)
            if len(boxes_in) == 0:
                return bgr,boxes
            box_shift = torch.FloatTensor([[int(shift_x),int(shift_y),int(shift_x),int(shift_y),0]]).expand_as(boxes_in)
            boxes_in = boxes_in+box_shift
            return after_shfit_image,boxes_in
        return bgr,boxes

    def getGroundTruth(self, coords):
        feature_size = self.input_size // self.grid_size
        ground_truth = np.zeros([feature_size, feature_size, 10 + self.class_num])

        for coord in coords:
            xmin, ymin, xmax, ymax, class_id = coord

            ground_width = xmax - xmin
            ground_high = ymax - ymin

            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2

            index_row = int(center_y / self.grid_size)
            index_col = int(center_x / self.grid_size)
            # ONE-HOT编码
            class_one_hot = self.One_Hot(class_id)
            # 编码真实标签
            ground_truth[index_row][index_col][:5] = [(center_x / self.grid_size) - index_col,
                                                      (center_y / self.grid_size) - index_row,
                                                      ground_width / self.input_size,
                                                      ground_high / self.input_size,
                                                      1.0]
            # ground_truth[index_row][index_col][5:10] = [xmin, ymin, xmax, ymax, (ymax-ymin)*(xmax-xmin)]
            ground_truth[index_row][index_col][5:10] = [(center_x / self.grid_size) - index_col,
                                                      (center_y / self.grid_size) - index_row,
                                                      ground_width / self.input_size,
                                                      ground_high / self.input_size,
                                                      1.0]
            ground_truth[index_row][index_col][10:30] = class_one_hot
        return ground_truth





if __name__=='__main__':
    ground_truth_test = VocDataset(img_path, img_annotations, CLASSES, is_train=True, class_num=20,
                 label_smooth_value=0.05, input_size=448, grid_size=64)
    for i in range(19353):
        img, truth = ground_truth_test.__getitem__(i)










