import pickle
import random

with open('/home/cxu-serve/p1/zli82/dataset/A2D/Annotations/annotated_neighbor_4_poly_box_by_cls.pkl', 'rb') as f:
    original_pkl = pickle.load(f)

modes = ['actor', 'action', 'actor_action']


val_set = {}
test_set = {}

val_img_set = set()
test_img_set = set()

complete_img_set = False

for mode in modes:
    lst = original_pkl[mode]
    val_lst = []
    test_lst = []
    for cls_records in lst:
        if len(cls_records) == 0:
            val_lst.append([])
            test_lst.append([])
        else:
            split_idx = int(len(cls_records) / 2)
            if not complete_img_set:
                random.shuffle(cls_records)
                candidate_val_set = set([record['img'] for record in cls_records[:split_idx]])
                val_img_set = val_img_set.union(candidate_val_set - test_img_set)
                candidate_test_set = set([record['img'] for record in cls_records[split_idx:]])
                test_img_set = test_img_set.union(candidate_test_set - val_img_set)
            val_lst.append([])
            test_lst.append([])
            for record in cls_records:
                if record['img'] in val_img_set:
                    val_lst[-1].append(record)
                elif record['img'] in test_img_set:
                    test_lst[-1].append(record)
                else:
                    raise RuntimeError('not exists in img set!')

    complete_img_set = True
    assert len(lst) == len(val_lst) == len(test_lst)

    val_set[mode] = val_lst
    test_set[mode] = test_lst

with open('/home/cxu-serve/p1/zli82/dataset/A2D/Annotations/small_val_box_by_cls.pkl', 'wb') as val_f, \
     open('/home/cxu-serve/p1/zli82/dataset/A2D/Annotations/small_test_box_by_cls.pkl', 'wb') as test_f, \
     open('/home/cxu-serve/p1/zli82/dataset/A2D/list/small_a2d_for_249/val.txt', 'w') as val_lst_f, \
     open('/home/cxu-serve/p1/zli82/dataset/A2D/list/small_a2d_for_249/test.txt', 'w') as test_lst_f:
    pickle.dump(val_set, val_f)
    pickle.dump(test_set, test_f)
    val_lst_f.write('\n'.join(list(val_img_set)))
    test_lst_f.write('\n'.join(list(test_img_set)))
