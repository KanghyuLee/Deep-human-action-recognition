# Deep-human-action-recognition
This program is framework for 2D and 3D human pose estimation and action recognition.
This code is corrected errors in the original code and easily derive the results. In case of 2D Dataset, MPII, visualizing pose estimation result for single person.

![Github](https://user-images.githubusercontent.com/71116312/92901675-9f4c4b00-f45b-11ea-989c-e7116531289c.png)

# Requirement
Install required python packages as below:

    pip3 install -r requirements.txt


# How to run
## 1. Dataset
User have to download each dataset. This program need 2 datasets, [MPII](http://human-pose.mpi-inf.mpg.de/) and [NTU RGB+D](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp) dataset.
Then, locate each dataset in `Deep-human-action-recognition/datasets/MPII or NTU`.

In case of NTU, you have to run `extract-resize-videos.py` to extract frames from videos at first. To run `extract-resize-videos.py`, parameter which denote how many number of subjects in NTU dataset is used to train have to be input.

## 2. 2D pose estimation (MPII)
### Train
`train_pmii_singleperson.py` can train model, but you need to revise dataset directory in line 55. This process is not required because `eval_mpii_singleperson.py` downloads the weights that the author has already uploaded and run.

### Test
`eval_mpii_singleperson.py` is test code. If you train and have your own weight file, you have to change weight file path in line 30 and 31.

## 3. 3D pose estimation and action recognition (NTU RGB+D)
### Train
`train_ntu_spnet.py` train spnet which is consists of pyramid and scale layers in TPAMI2020. Before training, datapaths have to be changed in line 66 ~ 78. Trained model is saved in h5 format in `Deep-human-action-recognition/exp/ntu`.

### Test
`eval_ntu_multitask.py` is test file. Dataset and trained weight path should be changed in line 48 and 58. 

# Citation
@ARTICLE{Luvizon_2020_TPAMI,
  author={D. {Luvizon} and D. {Picard} and H. {Tabia}},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Multi-task Deep Learning for Real-Time 3D Human Pose Estimation and Action Recognition}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},
}

@InProceedings{Luvizon_2018_CVPR,
  author = {Luvizon, Diogo C. and Picard, David and Tabia, Hedi},
  title = {2D/3D Pose Estimation and Action Recognition Using Multitask Deep Learning},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2018}
}
