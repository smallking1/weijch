import numpy as np
import torch

def iou(bounding_box, ground_coord):
    boxes = np.zeros((4))
    boxes[0] = int(bounding_box[0] - bounding_box[2] / 2)
    boxes[1] = int(bounding_box[1] - bounding_box[3] / 2)
    boxes[2] = int(bounding_box[0] + bounding_box[2] / 2)
    boxes[3] = int(bounding_box[1] + bounding_box[3] / 2)
    choice = np.zeros((4))
    choice[0] = int(ground_coord[0] - ground_coord[2] / 2)
    choice[1] = int(ground_coord[1] - ground_coord[3] / 2)
    choice[2] = int(ground_coord[0] + ground_coord[2] / 2)
    choice[3] = int(ground_coord[1] + ground_coord[3] / 2)


    predict_Area = (boxes[2] - boxes[0]) * (boxes[3] - boxes[1])
    ground_Area = (choice[2] - choice[0]) * (choice[3] - choice[1])

    # 计算交集的面积，左边取大者，右边取小，上边取大，下边取小。

    CrossLX = max(boxes[0], choice[0])
    CrossRX = min(boxes[2], choice[2])
    CrossUY = max(boxes[1], choice[1])
    CrossDY = min(boxes[3], choice[3])

    if CrossRX < CrossLX or CrossDY < CrossUY:
        return 0

    inter_Area = (CrossRX - CrossLX) * (CrossDY - CrossUY)
    return inter_Area / (predict_Area + ground_Area - inter_Area)




def nms(bounding_boxes, confidence_threshold, iou_threshold):
    # boxRow:x, y, dx, dy, c
    # 1.初步筛选，置信度较高的那个
    boxes = []
    for boxrow in bounding_boxes:
        if boxrow[4].item() < confidence_threshold and boxrow[9].item() < confidence_threshold:
            continue
        classes = boxrow[10:-1].to('cpu').detach().numpy()
        class_probality_index = np.argmax(classes, axis=-1)
        class_probality = classes[class_probality_index]
        print(class_probality)
        # 选择拥有置信度更大的box
        if boxrow[4].item() > boxrow[9].item():
            box = boxrow[0:4].to('cpu').detach().numpy()
        else:
            box = boxrow[5:9].to('cpu').detach().numpy()
        box = np.append(box, class_probality)
        box = np.append(box, class_probality_index)
        boxes.append(box)
    boxes = torch.tensor(sorted(boxes, key=(lambda x: [x[4]]), reverse=True))


    # 2.循环直到待筛选的box集合为空
    predicted_boxed = []
    while len(boxes) != 0:
        choiced_box = boxes[0]
        predicted_boxed.append(choiced_box)
        class_id = choiced_box[5]
        mask = torch.ByteTensor(boxes.size()[0]).bool()
        mask.zero_()
        for index in range(len(boxes)):
            if iou(boxes[index], choiced_box) < iou_threshold and boxes[index][5] != class_id :
                mask[index] = True
        boxes = boxes[mask]
    return predicted_boxed
