import os
import sys
import cv2
import numpy as np

if os.path.realpath(os.getcwd()) != os.path.dirname(os.path.realpath(__file__)):
    sys.path.append(os.getcwd())


from keras.models import Model
from keras.layers import concatenate
from keras.utils.data_utils import get_file

from deephar.config import mpii_sp_dataconf

from deephar.data import MpiiSinglePerson
from deephar.data import BatchLoader

from deephar.models import reception
from deephar.utils import *

sys.path.append(os.path.join(os.getcwd(), 'exp/common'))
from exp.common.mpii_tools import eval_singleperson_pckh

sys.path.append(os.path.join(os.getcwd(), 'datasets'))
import datasets.annothelper as annothelper

annothelper.check_mpii_dataset()

weights_file = 'weights_PE_MPII_cvpr18_19-09-2017.h5'
TF_WEIGHTS_PATH = \
        'https://github.com/dluvizon/deephar/releases/download/v0.1/' \
        + weights_file
md5_hash = 'd6b85ba4b8a3fc9d05c8ad73f763d999'

logdir = './'
if len(sys.argv) > 1:
    logdir = sys.argv[1]
    mkdir(logdir)
    sys.stdout = open(str(logdir) + '/log.txt', 'w')

"""Architecture configuration."""
num_blocks = 8
batch_size = 24
input_shape = mpii_sp_dataconf.input_shape
num_joints = 16

model = reception.build(input_shape, num_joints, dim=2,
        num_blocks=num_blocks, num_context_per_joint=2, ksize=(5, 5),
        concat_pose_confidence=False)

"""Load pre-trained model."""
weights_path = get_file(weights_file, TF_WEIGHTS_PATH, md5_hash=md5_hash,
        cache_subdir='models')
model.load_weights(weights_path)

"""Merge pose and visibility as a single output."""
outputs = []
for b in range(int(len(model.outputs) / 2)):
    outputs.append(concatenate([model.outputs[2*b], model.outputs[2*b + 1]],
        name='blk%d' % (b + 1)))
model = Model(model.input, outputs, name=model.name)

"""Load the MPII dataset."""
mpii = MpiiSinglePerson('../../datasets/MPII', dataconf=mpii_sp_dataconf)

"""Pre-load validation samples and generate the eval. callback."""
mpii_val = BatchLoader(mpii, x_dictkeys=['frame'],
        y_dictkeys=['pose', 'afmat', 'headsize'], mode=VALID_MODE,
        batch_size=mpii.get_length(VALID_MODE), num_predictions=1,
        shuffle=False)
printcn(OKBLUE, 'Pre-loading MPII validation data...')


webcam = cv2.VideoCapture(0)
while True:
    ret, frame = webcam.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    x = int((len(img[0]) - len(img)) / 2)
    w = int(len(img))
    y = 0
    h = int(len(img))
    img = img[y:y + h, x:x + w]

    #resize to 256x256
    img_256 = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
    #normalize
    img_256_norm = cv2.normalize(img_256, None, -1, 1, cv2.NORM_MINMAX, cv2.CV_64F)

    prediction = model.predict(np.expand_dims(img_256_norm, 0))

    pred_x_y_z_1 = prediction[5][0]

    pred_x_y_1 = pred_x_y_z_1[:, 0:2]
    pred_x_y_1080 = np.interp(pred_x_y_1, (0, 1), (0, 480))

    index = 0
    for x in pred_x_y_1080:
        img_out = cv2.circle(img, (int(x[0]), int(x[1])), 6, (255, 0, 0), -1)
        index = index + 1
        height, width, layers = img_out.shape
        size = (width, height)

        frame = None


    # Wirbelsäule
    x1 = int(pred_x_y_1080[3][0])
    y1 = int(pred_x_y_1080[3][1])

    x2 = int(pred_x_y_1080[2][0])
    y2 = int(pred_x_y_1080[2][1])

    img_out = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    x1 = int(pred_x_y_1080[2][0])
    y1 = int(pred_x_y_1080[2][1])

    x2 = int(pred_x_y_1080[1][0])
    y2 = int(pred_x_y_1080[1][1])

    img_out = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    x1 = int(pred_x_y_1080[1][0])
    y1 = int(pred_x_y_1080[1][1])

    x2 = int(pred_x_y_1080[0][0])
    y2 = int(pred_x_y_1080[0][1])

    img_out = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # l shoulder - neck - r shoulder
    x1 = int(pred_x_y_1080[4][0])
    y1 = int(pred_x_y_1080[4][1])

    x2 = int(pred_x_y_1080[1][0])
    y2 = int(pred_x_y_1080[1][1])

    img_out = cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    x1 = int(pred_x_y_1080[1][0])
    y1 = int(pred_x_y_1080[1][1])

    x2 = int(pred_x_y_1080[5][0])
    y2 = int(pred_x_y_1080[5][1])

    img_out = cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # linker Arm
    x1 = int(pred_x_y_1080[4][0])
    y1 = int(pred_x_y_1080[4][1])

    x2 = int(pred_x_y_1080[6][0])
    y2 = int(pred_x_y_1080[6][1])

    img_out = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    x1 = int(pred_x_y_1080[6][0])
    y1 = int(pred_x_y_1080[6][1])

    x2 = int(pred_x_y_1080[8][0])
    y2 = int(pred_x_y_1080[8][1])

    img_out = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # right Arm
    x1 = int(pred_x_y_1080[5][0])
    y1 = int(pred_x_y_1080[5][1])

    x2 = int(pred_x_y_1080[7][0])
    y2 = int(pred_x_y_1080[7][1])

    img_out = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    x1 = int(pred_x_y_1080[7][0])
    y1 = int(pred_x_y_1080[7][1])

    x2 = int(pred_x_y_1080[9][0])
    y2 = int(pred_x_y_1080[9][1])

    img_out = cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Pelvis
    x1 = int(pred_x_y_1080[10][0])
    y1 = int(pred_x_y_1080[10][1])

    x2 = int(pred_x_y_1080[0][0])
    y2 = int(pred_x_y_1080[0][1])

    img_out = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    x1 = int(pred_x_y_1080[0][0])
    y1 = int(pred_x_y_1080[0][1])

    x2 = int(pred_x_y_1080[11][0])
    y2 = int(pred_x_y_1080[11][1])

    img_out = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # linkes bein
    x1 = int(pred_x_y_1080[10][0])
    y1 = int(pred_x_y_1080[10][1])

    x2 = int(pred_x_y_1080[12][0])
    y2 = int(pred_x_y_1080[12][1])

    img_out = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    x1 = int(pred_x_y_1080[12][0])
    y1 = int(pred_x_y_1080[12][1])

    x2 = int(pred_x_y_1080[14][0])
    y2 = int(pred_x_y_1080[14][1])

    img_out = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # Rechtes Bein
    x1 = int(pred_x_y_1080[11][0])
    y1 = int(pred_x_y_1080[11][1])

    x2 = int(pred_x_y_1080[13][0])
    y2 = int(pred_x_y_1080[13][1])

    img_out = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    x1 = int(pred_x_y_1080[13][0])
    y1 = int(pred_x_y_1080[13][1])

    x2 = int(pred_x_y_1080[15][0])
    y2 = int(pred_x_y_1080[15][1])

    img_out = cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    #Save and Show Frame
    img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    # img_array.append(img_out)
    cv2.imshow('Recording KINECT Video Stream', img_out)







    key = cv2.waitKey(1)
    if key == 27:  # esc
        break
    elif key == 112:  # 'p'
        if delay == 33:
            delay = 0
        else:
            delay = 33

# [x_val], [p_val, afmat_val, head_val] = mpii_val[0]
#
# eval_singleperson_pckh(model, x_val, p_val[:,:,0:2], afmat_val, head_val)

