import os

root_path = '/home/ispl-ex39/Downloads/deephar-master/datasets'
ntu_root_path = '/home/ispl-ex39/hdd_ext/hdd2000'

def datasetpath(dataname):

    if dataname == 'NTU':
        data_path = os.path.join(ntu_root_path, dataname)
    else:
        data_path = os.path.join(root_path, dataname)

    return data_path