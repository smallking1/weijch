import cv2
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

data_path = r'D:\dataset\VOCdevkit\VOC2007\JPEGImages'

def show_labels_img(imgname):
    """imgname是图象下标"""
    img = cv2.imread(data_path + '/' + imgname + '.jpg')
    h, w = img.shape[:2]
    print(w,h)
    print(w,h)
    label = []
    with open(r'D:\dataset\VOCdevkit\VOC2007\label/' + imgname + '.txt','r') as f:
        for label in f:
            label = label.split(' ')
            label = [float(x.strip()) for x in label]
            pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))
            pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))
            cv2.putText(img, CLASSES[int(label[0])], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
            cv2.rectangle(img, pt1, pt2, (0, 0, 255, 2))

    cv2.imshow('img', img)
    cv2.waitKey(0)

show_labels_img('000002')
