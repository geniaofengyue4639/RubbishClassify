from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage import io


class MyDataset(Dataset):
    # 构造函数设置默认参数
    def __init__(self, txt, transform=None, target_transform=None):
        with open(txt, 'r') as fh:
            image = []
            for line in fh:
                line = line.strip('\n')  # 移除字符串首尾的换行符
                line = line.rstrip()  # 删除末尾空
                words = line.split()  # 以空格为分隔符 将字符串分成
                image.append((words[0], int(words[1])))  # image中包含有图像路径和标签
        self.image = image
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.image[index]
        # 调用定义的loader方法
        img = io.imread('./garbage/' + fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image)


train_data = MyDataset(txt='./garbage/train_list.txt', transform=transforms.Compose([transforms.ToPILImage(),
                                                                                     transforms.Resize(size=(224, 224)),
                                                                                     transforms.ToTensor()]))
test_data = MyDataset(txt='./garbage/test_list.txt', transform=transforms.Compose([transforms.ToPILImage(),
                                                                                   transforms.Resize(size=(224, 224)),
                                                                                   transforms.ToTensor()]))
valid_data = MyDataset(txt='./garbage/validate_list.txt', transform=transforms.Compose([transforms.ToPILImage(),
                                                                                        transforms.Resize(
                                                                                            size=(224, 224)),
                                                                                        transforms.ToTensor()]))

# train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
train_loader = DataLoader(dataset=train_data, batch_size=30, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=30)
valid_loader = DataLoader(dataset=valid_data, batch_size=30)
