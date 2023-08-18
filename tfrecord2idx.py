from subprocess import call
import os.path
from os import listdir
from os.path import isfile, join


tfrecord_dir = "imagenet_tfrecord/train"
idx_dir = "idx_files"
onlyfiles = [f for f in listdir(tfrecord_dir) if isfile(join(tfrecord_dir, f))]

batch_size = 16
tfrecord2idx_script = "tfrecord2idx"

if not os.path.exists(idx_dir):
    os.mkdir(idx_dir)

for file in onlyfiles:
    tfrecord_idx = join(idx_dir, file+".idx")
    if not os.path.isfile(tfrecord_idx):
        call([tfrecord2idx_script, join(tfrecord_dir, file), tfrecord_idx])
