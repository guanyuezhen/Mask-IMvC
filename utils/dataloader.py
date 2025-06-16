import numpy as np
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler
from utils.get_indicator_matrix_A import get_mask


def get_data(data_name, miss_rate, mask_seed):
    # load the datasets
    data_path = "./datasets/" + data_name + ".mat"
    data = sio.loadmat(data_path)
    if data['fea'].shape[1] < data['fea'].shape[0]:
        data_x = data['fea'][0]
    else:
        data_x = data['fea'][:][0]
    data_y = data['gt'].flatten()
    # get the basic information of the datasets
    view_num = data_x.shape[0]
    sample_num = data_x[0].shape[0]
    cluster_num = len(np.unique(data_y))
    input_dims = [data_x[v].shape[1] for v in range(view_num)]
    # generate missing or observed mask
    random_sequence = np.random.permutation(sample_num)
    mask = get_mask(view_num, sample_num, miss_rate, mask_seed)
    # random permutation the sample orders
    for v in range(view_num):
        data_x[v] = data_x[v][random_sequence]
    data_y = data_y[random_sequence]
    mask = mask[random_sequence]
    # normalize the datasets
    for v in range(view_num):
        pipeline = MinMaxScaler()
        data_x[v] = pipeline.fit_transform(data_x[v])
    print(
        f"Data: {data_name},"
        f" number of data: {sample_num},"
        f" views: {view_num},"
        f" clusters: {cluster_num},"
        f" dims of each view: {input_dims}")

    return mask, data_x, data_y, view_num, sample_num, cluster_num, input_dims
