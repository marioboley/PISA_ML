"""
This algorithm is for calculate single variable with target variable MI
"""
from binning import *
from permutation import *

class VariableMI:

    def __init__(self, x, y, xbin=False, ybin=False, bin_type='freq', x_bin=2, y_bin=2,
                 permut=False, fraction=True): # self.estimator = estimator
        # self.estimator = estimator
        self.x = x
        self.y = y
        self.bin_type = bin_type
        self.xbin = xbin
        self.ybin = ybin
        self.x_bin = x_bin
        self.y_bin = y_bin
        self.permut = permut
        self.fraction = fraction

    def summary(self):
        if self.ybin:  # y bins
#             if self.y_bin == 0:
#                 return 0
            # we don't discrete y
            y_freq = equal_freq(self.y, 8) \
                if self.bin_type == '' \
                else equal_width(self.y, self.y_bin)
        else:
            y_freq = self.y

        x_features = self.x
        if self.xbin:  # x bins
            if self.x_bin == 0:
                return 0
            if self.bin_type == 'freq':
                self.x_bin -= 1
                new_freq = x_features
                candidate = None
                for _ in range(self.x_bin):
                    candi = 0
                    new_freq = None
                    # x_data is previous binning using for loop, list of x_features, should find the best MI
                    # x_freq = frequency list
                    x_data_list, x_freq_list = equal_freq(x_features, self.x_bin) # refine_equal_freq1d(orgdata=x_features, modi_data=candidate)
                    for i in range(len(x_freq_list)):
                        mi = fGain(x_freq_list[i], y_freq)
                        if mi > candi:
                            candi = mi
                            candidate = x_data_list[i]
                            new_freq = x_freq_list[i]
                x_freq = new_freq  # get last x_freq
            else:
                x_freq = equal_width(x_features, self.x_bin)
        else:
            x_freq = x_features
        mi = fGain(x_freq, y_freq) if self.fraction else gain(x_freq, y_freq)
        if self.permut:
            permut = Permutation(x_freq, y_freq).summary()
            permut = permut / entropy(y_freq) if self.fraction else permut
            mi -= permut
        return mi
