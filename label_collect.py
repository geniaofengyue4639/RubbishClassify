from DataLodaer import valid_data, train_data, test_data
import numpy as np
from garbage_eval import garbage_class


class Nstr:
    def __init__(self, arg):
        self.x = arg

    def __sub__(self, other):
        c = self.x.replace(other.x, "")
        return c


classify_list = garbage_class()


def labelpath_cut(path_in):
    path1 = Nstr(path_in)
    path2 = Nstr('C:/Users/99538/Desktop/rubbish/garbage/')
    path_name = path1 - path2
    valid_data.image = np.array(valid_data.image)
    for i in valid_data.image:
        if i[0] == path_name:
            realabel = classify_list[str(i[1])]
            return realabel
    for i in train_data.image:
        if i[0] == path_name:
            realabel = classify_list[str(i[1])]
            return realabel
    for i in test_data.image:
        if i[0] == path_name:
            realabel = classify_list[str(i[1])]
            return realabel

    return '无此图片标签'


