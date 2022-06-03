import sys

import numpy as np
import torch
import torch.nn as nn
import math



class YOLO_V1_LOSS(nn.Module):
    # 有物体的损失权重为l_coord， 无物体的损失权重为l_noobj.
    def __init__(self, S=7, B=2, classes=20, l_coord=5, l_noobj=0.5, epoch_threshold=400):
        super(YOLO_V1_LOSS, self).__init__()
        self.S = S
        self.B = B
        self.classes = classes
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.epoch_threshold = epoch_threshold

    def iou(self,bounding_box, ground_coord, gridX, gridY, img_size=448, grid_size=64):
        # 将先验框返回原图大小
        predict_box = np.array([0, 0, 0, 0])
        predict_box[0] = int((bounding_box[0] + gridX) * grid_size)
        predict_box[1] = int((bounding_box[1] + gridY) * grid_size)
        predict_box[2] = int(bounding_box[2] * img_size)
        predict_box[3] = int(bounding_box[3] * img_size)

        # predict_box[0] = float((bounding_box[0]/14) - 0.5 * bounding_box[3])
        # predict_box[1] = float((bounding_box[1]/14) - 0.5 * bounding_box[4])
        # predict_box[2] = float((bounding_box[0]/14) + 0.5 * bounding_box[3])
        # predict_box[3] = float((bounding_box[1]/14) + 0.5 * bounding_box[4])


        # 将xywh改为xmin ymin xmax ymin
        predict_coord = [max(0, predict_box[0] - predict_box[2] / 2),
                         max(0, predict_box[1] - predict_box[3] / 2),
                         min(img_size - 1, predict_box[0] + predict_box[2] / 2),
                         min(img_size - 1, predict_box[1] + predict_box[3] / 2)]

        # ground_truth_box = np.array([0, 0, 0, 0])
        # predict_box[0] = int((bounding_box[0] + gridX) * grid_size)
        # predict_box[1] = int((bounding_box[1] + gridY) * grid_size)
        # predict_box[2] = int(bounding_box[2] * img_size)
        # predict_box[3] = int(bounding_box[3] * img_size)

        # ground_truth_box[0] = float((ground_coord[0] / 14) - 0.5 * ground_coord[3])
        # ground_truth_box[1] = float((ground_coord[1] / 14) - 0.5 * ground_coord[4])
        # ground_truth_box[2] = float((ground_coord[0] / 14) + 0.5 * ground_coord[3])
        # ground_truth_box[3] = float((ground_coord[1] / 14) + 0.5 * ground_coord[4])

        predict_Area = (predict_coord[2] - predict_coord[0]) * (predict_coord[3] - predict_coord[1])

        ground_Area = ground_coord[9]
        # predict_Area = (predict_box[2] - predict_box[0]) * (predict_box[3] - predict_box[1])
        # ground_Area = (ground_truth_box[2] - ground_truth_box[0]) * (ground_truth_box[3] - ground_truth_box[1])

        # 计算交集的面积，左边取大者，右边取小，上边取大，下边取小。


        CrossLX = max(predict_coord[0], ground_coord[5])
        CrossRX = min(predict_coord[2], ground_coord[7])
        CrossUY = max(predict_coord[1], ground_coord[6])
        CrossDY = min(predict_coord[3], ground_coord[8])

        # CrossLX = max(predict_box[0], ground_truth_box[0])
        # CrossRX = min(predict_box[2], ground_truth_box[2])
        # CrossUY = max(predict_box[1], ground_truth_box[1])
        # CrossDY = min(predict_box[3], ground_truth_box[3])

        if CrossRX < CrossLX or CrossDY < CrossUY:
            return 0

        inter_Area = (CrossRX - CrossLX) * (CrossDY - CrossUY)
        # print(("inter_Area={} predict_Area={} ground_Area={} ").format(inter_Area,predict_Area,ground_Area))
        # print(inter_Area / (predict_Area + ground_Area - inter_Area))
        return inter_Area / (predict_Area + ground_Area - inter_Area)

    def forward(self, bounding_boxes, ground_truth):
        # 定义三个计算损失的变量 正样本定位损失 样本置信度损失 样本类别损失
        loss = 0
        loss_coord = 0
        loss_confidence = 0
        loss_classes = 0
        iou_sum = 0
        object_num = 0

        mseloss = nn.MSELoss(size_average=False)
        for batch in range(len(bounding_boxes)):
            # 先行后列
            for indexRow in range(self.S):
                for indexCol in range(self.S):
                    bounding_box = bounding_boxes[batch][indexRow][indexCol]
                    predict_one = bounding_box[:5]
                    predict_two = bounding_box[5:10]
                    ground_box = ground_truth[batch][indexRow][indexCol]
                    # 如果此处不存在ground_box，则为背景，两个预测框都为负样本。
                    if round(ground_box[4].item()) == 0:
                        loss = loss + self.l_noobj * (torch.pow(predict_one[4], 2) + torch.pow(predict_two[4], 2))
                        loss_confidence += self.l_noobj * (math.pow(predict_one[4], 2) + math.pow(predict_two[4], 2))
                        # loss = loss + self.l_noobj * (mseloss(predict_one[4], torch.zeros_like(predict_one[4])) + mseloss(predict_two[4],torch.zeros_like(predict_two[4])))
                        # loss_confidence += loss + self.l_noobj * (mseloss(predict_one[4], torch.zeros_like(predict_one[4])) + mseloss(predict_two[4],torch.zeros_like(predict_two[4])))
                    else:
                        object_num += 1
                        predict_one_iou = self.iou(predict_one, ground_box, indexCol, indexRow)
                        predict_two_iou = self.iou(predict_two, ground_box, indexCol, indexRow)
                        # 让两个预测的box与ground box 的iou大的作为正样本，另一个作为负样本。
                        if predict_one_iou > predict_two_iou:
                            predict_box = predict_one
                            iou = predict_one_iou
                            no_predict_box = predict_two
                        else:
                            predict_box = predict_two
                            iou = predict_two_iou
                            no_predict_box = predict_one
                        # 定位损失
                        loss = loss + self.l_coord * (torch.pow((predict_box[0] - ground_box[0]), 2)+
                                                      torch.pow((predict_box[1] - ground_box[1]), 2)+
                                                      torch.pow((torch.sqrt(predict_box[2]) - torch.sqrt(ground_box[2]))
                                                                ,2) + torch.pow((torch.sqrt(predict_box[3]) -
                                                                                 torch.sqrt(ground_box[3])), 2))

                        loss_coord += self.l_coord * (math.pow((predict_box[0] - ground_box[0]), 2)+
                                                      math.pow((predict_box[1] - ground_box[1]), 2)+
                                                      math.pow((torch.sqrt(predict_box[2]) - math.sqrt(ground_box[2]))
                                                                ,2) + math.pow((math.sqrt(predict_box[3]) -
                                                                                math.sqrt(ground_box[3])), 2))

                        # loss = loss + self.l_coord * (mseloss(predict_box[0], ground_box[0]) +
                        #                               mseloss(predict_box[1], ground_box[1]) +
                        #                               mseloss(torch.sqrt(predict_box[2]), torch.sqrt(ground_box[2]))
                        #                                         + mseloss(torch.sqrt(predict_box[3]),torch.sqrt(ground_box[3])))
                        #
                        # loss_coord += self.l_coord * (mseloss(predict_box[0], ground_box[0]) +
                        #                               mseloss(predict_box[1], ground_box[1]) +
                        #                               mseloss(torch.sqrt(predict_box[2]), torch.sqrt(ground_box[2]))
                        #                                         + mseloss(torch.sqrt(predict_box[3]),torch.sqrt(ground_box[3])))

                        # 置信度损失(正样本)
                        loss = loss + torch.pow((predict_box[4] - iou), 2)
                        loss_confidence += math.pow((predict_box[4] - iou), 2)
                        # loss = loss + mseloss(predict_box[4], iou * torch.ones_like(predict_box[4]))
                        # loss_confidence += mseloss(predict_box[4], iou * torch.ones_like(predict_box[4]))
                        iou_sum += iou

                        # 分类损失
                        ground_class = ground_box[10:]
                        predict_class = bounding_box[self.B * 5:]
                        loss = loss + mseloss(ground_class, predict_class)
                        loss_classes += mseloss(ground_class, predict_class)

                        # 置信度损失(负样本)

                        loss = loss + self.l_noobj * torch.pow((no_predict_box[4] - 0), 2)
                        loss_confidence += self.l_noobj * math.pow((no_predict_box[4] - 0), 2)
                        # loss = loss + self.l_noobj * mseloss(no_predict_box[4], torch.zeros_like(no_predict_box[4]))
                        # loss_confidence +=  self.l_noobj * mseloss(no_predict_box[4], torch.zeros_like(no_predict_box[4]))


        return loss,  loss_coord, loss_confidence,  loss_classes,  iou_sum, object_num

























































