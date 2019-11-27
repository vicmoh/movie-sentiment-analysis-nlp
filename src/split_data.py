# File for moving the 15% of the data
# to test folder for testing

import os
import shutil
path_pos = './assets/review_polarity/data/pos/'
moveTo_pos = './assets/review_polarity/test/pos/'
path_neg = './assets/review_polarity/data/neg/'
moveTo_neg = './assets/review_polarity/test/neg/'


def moveFiles(path, moveTo):
    files = os.listdir(path)
    count = 0
    for f in files:
        count += 1
        src = path+f
        dst = moveTo+f
        shutil.move(src, dst)
        if count == 150:
            break


# Move files for the pos data
moveFiles(path_pos, moveTo_pos)
# Move files for the neg data
moveFiles(path_neg, moveTo_neg)
