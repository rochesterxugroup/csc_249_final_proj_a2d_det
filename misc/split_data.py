import random
import os

train_fpath = '/home/cxu-serve/p1/zli82/dataset/A2D/list/train_annotated_neighbor_4_poly.txt'
test_fpath = '/home/cxu-serve/p1/zli82/dataset/A2D/list/test_annotated_neighbor_4_poly.txt'

with open(train_fpath) as train_val_f, open(test_fpath) as test_f:
    train_val_lst = train_val_f.readlines()
    test_lst = test_f.readlines()

random.shuffle(train_val_lst)
random.shuffle(test_lst)


train_lst, val_lst = train_val_lst[:4750], train_val_lst[4750 : 4750 + 1583]
test_lst = test_lst[:1583]

root = '/home/cxu-serve/p1/zli82/dataset/A2D/list/small_a2d_for_249'

train_save_path = os.path.join(root, 'train.txt')
val_save_path = os.path.join(root, 'val.txt')
test_save_path = os.path.join(root, 'test.txt')

with open(train_save_path, 'w+') as train_save_f, \
     open(val_save_path, 'w+') as val_save_f, \
     open(test_save_path, 'w+') as test_save_f:
    train_save_f.write(''.join(train_lst))
    val_save_f.write(''.join(val_lst))
    test_save_f.write(''.join(test_lst))
