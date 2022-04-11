from DataLodaer import *
import torch
import torchvision
import torch.nn as nn
import json
from torchvision import transforms


def evalvation(image):
    # GPU选择
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 处理图片
    in_put = transforms.Compose([transforms.ToPILImage(), transforms.Resize(size=(224, 224)),
                                 transforms.ToTensor()])
    image = in_put(image)
    # 加载4个预训练模型
    pretrained_alexnet = torchvision.models.alexnet(pretrained=True)
    pretrained_alexnet.classifier = nn.Sequential(nn.Dropout(),
                                                  nn.Linear(256 * 6 * 6, 4096),
                                                  nn.ReLU(inplace=True),
                                                  nn.Dropout(),
                                                  nn.Linear(4096, 4096),
                                                  nn.ReLU(inplace=True),
                                                  nn.Linear(4096, 40))

    pretrained_googlenet = torchvision.models.googlenet(pretrained=True)
    pretrained_googlenet.fc = nn.Linear(1024, 40)

    pretrained_vggnet = torchvision.models.vgg11(pretrained=True)
    pretrained_vggnet.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 40),
    )

    pretrained_resnet = torchvision.models.resnet18(pretrained=True)
    pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, 40)

    # 给模型加载训练后的权重
    model_alexnet = pretrained_alexnet
    model_alexnet.load_state_dict(torch.load('AlexNet.pt'))

    model_googlenet = pretrained_googlenet
    model_googlenet.load_state_dict(torch.load('GoogLeNet.pt'))

    model_vggnet = pretrained_vggnet
    model_vggnet.load_state_dict(torch.load('VGG.pt'))

    model_resnet = pretrained_resnet
    model_resnet.load_state_dict(torch.load('ResNet.pt'))

    # 建立模型序列 ，准备预测
    model = [model_alexnet, model_googlenet, model_vggnet, model_resnet]

    result_list = []
    classify_list = garbage_class()

    # image, label = valid_data[100]
    for module in model:
        module.eval()
        module.to(device)
        with torch.no_grad():
            image = image.to(device)
            input_tensor = torch.unsqueeze(image, dim=0)  # 增加单个tensor维度，才能进入网络计算
            output = module(input_tensor)
            _, predicted = torch.max(output.data, 1)
            print('预测类别为:', predicted.item())
            # print('实际标签为', label)
            result_list.append(classify_list[str(predicted.item())])

    # result_list.append(classify_list[str(label)])

    return result_list


def garbage_class():
    with open('./garbage/garbage_classify_rule.json', encoding='utf-8') as json_file:
        reference_list = json.load(json_file)

    return reference_list
