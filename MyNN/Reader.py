# coding: utf-8


def read_data(file_name):
    """
    读取数据
    :return:
    """
    f = open(file_name)
    if not f:
        print "打开文件失败"
        return -1

    data = []
    lines = f.readlines()
    for line in lines:
        data.append(line.strip('\n').strip('(').strip(')').split(','))

    return data
