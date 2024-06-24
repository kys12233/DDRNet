import os


def op_file():
    # train
    train_image_root = 'image/train/'
    train_label_root = 'label/train/'
    train_image_path = 'data/drug/image/train'
    train_label_path = 'data/drug/label/train'

    trainImageList = os.listdir(train_image_path)
    trainLabelList = os.listdir(train_label_path)

    train_image_list = []
    for image in trainImageList:
        train_image_list.append(train_image_root + image)

    train_label_list = []
    for label in trainLabelList:
        train_label_list.append(train_label_root + label)

    train_list_path = 'data/list/drug/train.lst'
    file = open(train_list_path, 'w').close()
    with open(train_list_path, 'w', encoding='utf-8') as f:
        for i1, i2 in zip(train_image_list, train_label_list):
            print(i1, i2)
            f.write(i1 + "   " + i2 + "\n")
    f.close()

    # test
    test_image_root = 'image/test/'
    test_label_root = 'label/test/'
    test_image_path = 'data/drug/image/test'

    testImageList = os.listdir(test_image_path)

    test_image_list = []
    for image in testImageList:
        test_image_list.append(test_image_root + image)

    test_list_path = 'data/list/drug/test.lst'
    file = open(test_list_path, 'w').close()
    with open(test_list_path, 'w', encoding='utf-8') as f:
        for i1 in test_image_list:
            f.write(i1 + "\n")
    f.close()

    # val
    val_image_root = 'image/val/'
    val_label_root = 'label/val/'
    val_image_path = 'data/drug/image/val'
    val_label_path = 'data/drug/label/val'

    valImageList = os.listdir(val_image_path)
    valLabelList = os.listdir(val_label_path)

    val_image_list = []
    for image in valImageList:
        val_image_list.append(val_image_root + image)

    val_label_list = []
    for label in valLabelList:
        val_label_list.append(val_label_root + label)

    val_list_path = 'data/list/drug/val.lst'
    file = open(val_list_path, 'w').close()
    with open(val_list_path, 'w', encoding='utf-8') as f:
        for (i1, i2) in zip(val_image_list, val_label_list):
            f.write(i1 + "   " + i2 + "\n")
    f.close()

    # trainval
    trainval_list_path = 'data/list/drug/trainval.lst'
    file = open(trainval_list_path, 'w').close()
    with open(trainval_list_path, 'w', encoding='utf-8') as f:
        for (i1, i2) in zip(train_image_list, train_label_list):
            f.write(i1 + "   " + i2 + "\n")
    f.close()

    with open(trainval_list_path, 'a', encoding='utf-8') as f:
        for (i1, i2) in zip(val_image_list, val_label_list):
            f.write(i1 + "   " + i2 + "\n")
    f.close()


if __name__ == '__main__':
    op_file()