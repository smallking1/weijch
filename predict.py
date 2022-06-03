import cv2
import torch
from yolov1 import ResNet, Bottleneck
import torchvision.transforms as transforms
from tool.NMS import nms





def resize_image_with_coords(img, input_size=448):
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
    return new_image


CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

img_path = './test/000042.jpg'

model_path = './weights/YOLO_V1_50.pth'

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



transform_common = transforms.Compose([
            transforms.ToTensor(),  # height * width * channel -> channel * height * width
            transforms.Normalize(mean=(0.408, 0.448, 0.471), std=(0.242, 0.239, 0.234))  # 归一化后.不容易产生梯度爆炸的问题
        ])

if __name__ == "__main__":
    img = cv2.imread(img_path)
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(torch.load(model_path)['model'])
    model.to(device)
    model.eval()
    img = resize_image_with_coords(img)
    input = transform_common(img).unsqueeze(0).to(device)
    output = model(input)
    bounding_boxes = []
    # 遍历7*7恢复到448的中心与宽高
    for batch in range(output.shape[0]):
        for row in range(output.shape[1]):
            for col in range(output.shape[2]):

                bounding_box = output[batch][row][col]
                bounding_box[:4] = torch.tensor([int((bounding_box[0] + col) * 64),
                                    int((bounding_box[1] + row) * 64),
                                    int(bounding_box[2] * 448),
                                    int(bounding_box[3] * 448)])
                bounding_box[5:9] = torch.tensor([int((bounding_box[5] + col) * 64),
                                    int((bounding_box[6] + row) * 64),
                                    int(bounding_box[7] * 448),
                                    int(bounding_box[8] * 448)])
                bounding_boxes.append(bounding_box)
    pre_box = nms(bounding_boxes, 0.5, 0.5)
    for i, box in enumerate(pre_box):
        cv2.putText(img, CLASSES[int(box[5])], (int(box[0]-box[2]/2), int(box[1]-box[3]/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.rectangle(img, (int(box[0]-box[2]/2), int(box[1]-box[3]/2)),(int(box[0]+box[2]/2), int(box[1]+box[3]/2)),color=(255,0,0))

    cv2.imshow('test',img)
    cv2.waitKey(0)








