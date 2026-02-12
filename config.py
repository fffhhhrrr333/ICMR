import os
DEVICE='cuda'
EPOCHS=100
BATCH_SIZE=16
LR=0.01
ratio=0.5 
sample_num=3


MAXMIN=False
height,width = (256, 256)

ENCODER='resnet101'

WEIGHTS='imagenet'

name='CISRNet'

basedir=rf'./{name}/'

os.makedirs(basedir,exist_ok=True)
outptfile=basedir+f'{ENCODER}_{WEIGHTS}_{name}.pt'


loadstate=False
loadstateptfile=outptfile
def log(traintxt,ds):
    with open(traintxt,'a') as  f:
        f.write(ds)

import glob
import shutil
for file in glob.glob('./*.py'):
    os.makedirs(basedir+'code',exist_ok=True) 
    shutil.copyfile(file,basedir+'/'+'code/'+os.path.basename(file))

# --- Dataset and workspace paths (override before training if needed) ---
OPTICAL_DIR = "/train/opt"
RADAR_DIR = "/train/vv"
LABEL_DIR = "/train/flood_vv"
# Checkpoint and metrics output
CHECKPOINT_DIR = os.path.join(basedir, 'cCISR')
METRICS_CSV = os.path.join(basedir, 'mCISR.csv')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)



