import sys
sys.path.append("../")
import numpy as np
import utils.utilsgeo as ug
from scipy.spatial import distance


# run this file from ./datasets/ folder
# Creates datasets with larger K (max num neighbours) from the original datasets

def create_input(X_train, X_test, y_train, y_test, k, dataset):
    idx, dist = ug.create_sequence(X_train, X_test, int(k))
    idx_eucli, dist_eucli = ug.recovery_dist(idx[:X_train.shape[0], :],
                                             idx[X_train.shape[0]:, :], X_train[:, 2:],
                                             X_test[:, 2:], distance.euclidean)

    idx_geo = idx
    dist_geo = dist
    idx_eucli = idx_eucli
    dist_eucli = dist_eucli

    np.savez_compressed(f"./{dataset}/" + f'data{str(k)}.npz', dist_eucli=dist_eucli, dist_geo=dist_geo, idx_eucli=idx_eucli,
                        idx_geo=idx_geo, X_test=X_test, X_train=X_train, y_test=y_test, y_train=y_train)

if __name__ == "__main__":
    dataset = sys.argv[1]
    N = int(sys.argv[2])
    
    data = np.load(f"./{dataset}/data.npz", allow_pickle=True)

    # original data
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    create_input(X_train, X_test, y_train, y_test, N, dataset)