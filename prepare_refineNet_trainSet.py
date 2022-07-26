import shutil
import os
# src_path: the folder where save your multi-view images dataset
# dst_path: the destination path we want to put the images into as ground truth
# tgt_path: the folder includes randomly selected images, rendering by trained MoFaNeRF-coarse
src_path = "/data/myNerf/data/multiViewDataForNerf300New"
dst_path = "/data/MoFaNeRF_github/logs/mofanerf_0to300/renderonly_path_000000/rf_trainSet/gt"
tgt_path = "/data/MoFaNeRF_github/logs/mofanerf_0to300/renderonly_path_000000/rf_trainSet/train"

for c1 in os.listdir(tgt_path):
    for c2 in os.listdir(os.path.join(tgt_path, c1)):
        os.makedirs(os.path.join(dst_path, c1, c2), exist_ok=True)
        for c3 in os.listdir(os.path.join(tgt_path, c1, c2)):
            shutil.copy(os.path.join(src_path,c1,c2,c3),
                        os.path.join(dst_path,c1,c2,c3))
        print("done: ", os.path.join(dst_path, c1, c2))
