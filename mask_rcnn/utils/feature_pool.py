class FeatureRunningAvg:
    def __init__(self, num_class):
        self.avg_feat_lst = [None for _ in range(num_class)]
        self.weight_new = 0.1
        self.weight_old = 0.9

    def update(self, new_coming_feat, class_label: int):
        # assume that class label starts from 1
        assert class_label > 0
        if self.avg_feat_lst[class_label - 1] is None:
            self.avg_feat_lst[class_label - 1] = new_coming_feat
        else:
            self.avg_feat_lst[class_label - 1] = \
                self.weight_old * self.avg_feat_lst[class_label - 1] \
                + self.weight_new * new_coming_feat

    def obtain(self, class_label: int):
        assert class_label > 0
        return self.avg_feat_lst[class_label - 1]
