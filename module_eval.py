from DataLodaer import *
import torch
import torchvision
import torch.nn as nn


def module_eval():
    # GPU选择
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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

    for module in model:
        module.eval()
        module.to(device)
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = module(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            result = ('Test Accuracy of the model on the valid images: {} %'.format(100 * correct / total))
            result_list.append(result)

    return result_list
