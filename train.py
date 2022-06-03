import time
import sys
import torch
from tool.get_vocdata import VocDataset
from yolov1 import yolov1, ResNet, Bottleneck
from loss import YOLO_V1_LOSS
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import models
import os
import time
from yoloLoss import yoloLoss



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

B = 2
class_num = 20
batch_size = 4
lr = 0.001
lr_mul_factor_epoch_1 = 1.04
lr_epoch_2 = 0.0001
lr_epoch_77 = 0.00001
lr_epoch_107 = 0.000001

weight_decay = 0.0005
momentum = 0.9

epoch_val_loss_min = 1000000000
epoch_interval = 10

epoch_unfreeze = 10
epochs_num = 135

img_path = r'D:\dataset\VOCdevkit\VOC2007\JPEGImages'
img_annotations = r"D:\dataset\VOCdevkit\VOC2007\Annotations"
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']




# -----------------step 1: Dataset-----------------
# Dataset = VocDataset(img_path, img_annotations, CLASSES, is_train=True, class_num=20, input_size=448, grid_size=64)
# train_dataset, val_dataset = random_split(Dataset, [20000, 1503], generator=torch.Generator().manual_seed(0))

train_dataset = VocDataset(img_path, img_annotations, CLASSES, is_train=True, class_num=20, input_size=448, grid_size=64)
val_dataset = VocDataset(img_path, img_annotations, CLASSES, is_train=False, class_num=20, input_size=448, grid_size=64)

# -----------------step 2: Model-------------------
use_resnet = True
use_self_train = False
if use_resnet:
    YOlO = ResNet(Bottleneck, [3, 4, 6, 3])
    if use_self_train:
        YOlO.load_state_dict(torch.load('./weights/YOLO_V1_7.pth')['model'])
    else:
        resnet = models.resnet50(pretrained=True)
        new_state_dict = resnet.state_dict()
        dd = YOlO.state_dict()
        for k in new_state_dict.keys():
            if k in dd.keys() and not k.startswith('fc'):
                dd[k] = new_state_dict[k]
        YOlO.load_state_dict(dd)
    YOlO.to(device=device)
else:
    YOlO = yolov1(B, class_num)
    YOlO.yolo_weight_init()
    YOlO.to(device=device)

#---------------step3: LossFunction------------------
loss_function = YOLO_V1_LOSS().to(device=device)
# loss_function = yoloLoss(7,2,5,0.5).to(device=device)

#---------------step4: Optimizer---------------------
optimizer_SGD = torch.optim.SGD(YOlO.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

#---------------step4: Train-------------------------

if __name__ == "__main__":
    train_log_filename = "train_log.txt"
    train_log_filepath = os.path.join("./logs", train_log_filename)
    epoch = 0
    param_dict = {}
    while epoch <= epochs_num:
        epoch_train_loss = 0
        epoch_val_loss = 0
        epoch_train_iou = 0
        epoch_val_iou = 0
        epoch_train_object_num = 0
        epoch_val_object_num = 0
        epoch_train_loss_coord = 0
        epoch_val_loss_coord = 0
        epoch_train_loss_confidence = 0
        epoch_val_loss_confidence = 0
        epoch_train_loss_classes = 0
        epoch_val_loss_classes = 0
        total_loss = 0
        val_total_loss = 0

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
        train_len = train_loader.__len__()
        YOlO.train()
        with tqdm(total=train_len) as tbar:
            for batch_index, batch_train in enumerate(train_loader):

                train_data = batch_train[0].float().to(device=device)

                train_label = batch_train[1].float().to(device=device)

                loss = loss_function(YOlO(train_data), train_label)


                # total_loss += loss

                batch_loss = loss[0] / batch_size
                total_loss += batch_loss
                # # 定位损失
                # epoch_train_loss_coord = epoch_train_loss_coord + loss[1]
                # # 置信度损失
                # epoch_train_loss_confidence = epoch_train_loss_confidence + loss[2]
                # # 分类损失
                # epoch_train_loss_classes = epoch_train_loss_classes + loss[3]
                # # 交并比总和
                # epoch_train_iou = epoch_train_iou + loss[4]
                # # 训练目标对象个数
                # epoch_train_object_num = epoch_train_object_num + loss[5]

                optimizer_SGD.zero_grad()
                batch_loss.backward()
                optimizer_SGD.step()


                # batch_loss = float(batch_loss) * batch_size
                # epoch_train_loss = epoch_train_loss + batch_loss

                # 一个批量的总体loss
                # tbar.set_description(("train: coord_loss:{} confidence_loss:{} class_loss:{} avg_iou:{} batch_obj_num:{} lr:{}"
                #                       .format(round(float(loss[1]), 4), round(float(loss[2]), 4), round(float(loss[3]), 4),
                #                               round(float(loss[4] / loss[5]),4), loss[5], lr)), refresh=True)

                tbar.set_description( ("train: loss:{} average_loss:{}".format(batch_loss, (total_loss/ batch_index + 1))), refresh=True)

                tbar.update(1)

                # if epoch == 1:
                #     lr = min(lr * lr_mul_factor_epoch_1, lr_epoch_2)


            # 批量平均
            # loss_str = round(float(epoch_train_loss / train_len), 4)
            # loss_str_coord = round(float(epoch_train_loss_coord / train_len), 4)
            # loss_str_conf = round(float(epoch_train_loss_confidence / train_len), 4)
            # loss_class = round(float(epoch_train_loss_classes / train_len), 4)
            # avg_iou_str = round(float(epoch_train_iou / epoch_train_object_num), 4)




            #
            # train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}[coord_loss]{loss_str_coord}" \
            #                           "[confidence_loss]{loss_str_conf}[class_loss]{loss_class}[avg_iou]{avg_iou_str}\n"
            # to_write = train_log_txt_formatter.format(time_str=time.strftime("%Y_%m_%d_%H:%M:%S"),
            #                                           epoch = epoch,
            #                                           loss_str=" ".join(["{}".format(loss_str)]),
            #                                           loss_str_coord =" ".join(["{}".format(loss_str_coord)]),
            #                                           loss_str_conf=" ".join(["{}".format(loss_str_conf)]),
            #                                           loss_class=" ".join(["{}".format(loss_class)]),
            #                                           avg_iou_str=" ".join(["{}".format(avg_iou_str)]))
            # with open(train_log_filepath, "a") as f:
            #     f.write(to_write)
            # #
            # print("train-batch-mean loss:{} coord_loss:{} confidence_loss:{} class_loss:{} avg_iou:{} epoch_train_object_num:{}".format(
            #     loss_str, loss_str_coord,
            #     loss_str_conf,
            #     loss_class,
            #     avg_iou_str,epoch_train_object_num))

        val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=False)
        val_len = val_loader.__len__()
        YOlO.eval()
        with tqdm(total=val_len) as tbar:
            with torch.no_grad():
                for batch_index, batch_val in enumerate(val_loader):
                    val_data = batch_val[0].float().to(device=device)
                    val_label = batch_val[1].float().to(device=device)
                    loss = loss_function(YOlO(val_data), val_label)
                    # val_total_loss += loss

                    batch_loss = loss[0] / batch_size
                    val_total_loss += batch_loss
                    # epoch_val_loss_coord = epoch_val_loss_coord + loss[1]
                    # # 置信度损失
                    # epoch_val_loss_confidence = epoch_val_loss_confidence + loss[2]
                    # # 分类损失
                    # epoch_val_loss_classes = epoch_val_loss_classes + loss[3]
                    # # 交并比总和
                    # epoch_val_iou = epoch_val_iou + loss[4]
                    # # 测试目标对象个数
                    # epoch_val_object_num = epoch_val_object_num + loss[5]
                    #
                    # batch_loss = float(batch_loss) * batch_size
                    # epoch_val_loss = epoch_val_loss + batch_loss

                    # tbar.set_description(("val: coord_loss:{} confidence_loss:{} class_loss:{} avg_iou:{} obj:{}"
                    #                       .format(round(float(loss[1]), 4), round(float(loss[2]), 4), round(float(loss[3]), 4),
                    #                               round(float(loss[4] / loss[5]), 4), loss[5])),
                    #                      refresh=True)
                    tbar.set_description(("val: loss:{} average_loss:{} "
                                          .format( batch_loss, (val_total_loss/ batch_index + 1))), refresh=True)

                    tbar.update(1)

                # print("val-batch-mean loss:{} coord_loss:{} confidence_loss:{} class_loss:{} avg_iou:{}".format(
                #     round(float(epoch_val_loss / val_len), 4), round(float(epoch_val_loss_coord / val_len), 4),
                #     round(float(epoch_val_loss_confidence / val_len), 4), round(float(epoch_val_loss_classes / val_len), 4),
                #     round(float(epoch_val_iou / epoch_val_object_num), 4)))

        epoch = epoch + 1

        if epoch == 60:
            lr = lr_epoch_2
        elif epoch == 100:
            lr = lr_epoch_77
        elif epoch == 120:
            lr = lr_epoch_107

        # if epoch == epoch_unfreeze:
        #     model.set_freeze_by_idxs(YOlO, [0])

        for param_group in optimizer_SGD.param_groups:
            param_group["lr"] = lr

        if epoch_val_loss < epoch_val_loss_min:
            epoch_val_loss_min = epoch_val_loss
            optimal_dict = YOlO.state_dict()

        if epoch % epoch_interval == 0:
            param_dict['model'] = YOlO.state_dict()
            param_dict['optim'] = optimizer_SGD
            param_dict['epoch'] = epoch
            param_dict['optimal'] = optimal_dict
            param_dict['epoch_val_loss_min'] = epoch_val_loss_min
            torch.save(param_dict, './weights/YOLO_V1_' + str(epoch) + '.pth')

















