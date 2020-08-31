
import numpy as np


def create_npy(filename: str, dataset: np.array=None):
    data = np.load(f'{filename}')
    return np.concatenate((dataset, data))


if __name__ == "__main__":
    base_path = f'./dataset/filtered'
    mu_path = f'{base_path}/move_up'
    md_path = f'{base_path}/move_down'
    nm_path = f'{base_path}/no_movement'

    # patient 1
    move_down_ds = np.load(f'{md_path}/1_1.npy')
    for i in range(2, 20):
        move_down_ds = create_npy(f'{md_path}/1_{i}.npy', move_down_ds)

    move_up_ds = np.load(f'{mu_path}/1_1.npy')
    for i in range(2, 16):
        move_up_ds = create_npy(f'{mu_path}/1_{i}.npy', move_up_ds)

    no_movement_ds = np.load(f'{nm_path}/1_1.npy')
    for i in range(2, 3):
        no_movement_ds = create_npy(f'{nm_path}/1_{i}.npy', no_movement_ds)

    # patient 2
    for i in range(1, 16):
        move_up_ds = create_npy(f'{mu_path}/2_{i}.npy', move_up_ds)

    for i in range(1, 16):
        move_down_ds = create_npy(f'{md_path}/2_{i}.npy', move_down_ds)

    for i in range(1, 14):
        no_movement_ds = create_npy(f'{nm_path}/2_{i}.npy', no_movement_ds)

    # patient 3
    for i in range(1, 15):
        move_up_ds = create_npy(f'{mu_path}/3_{i}.npy', move_up_ds)

    for i in range(1, 18):
        move_down_ds = create_npy(f'{md_path}/3_{i}.npy', move_down_ds)

    for i in range(1, 7):
        no_movement_ds = create_npy(f'{nm_path}/3_{i}.npy', no_movement_ds)

    # patient 4
    for i in range(2, 21):
        move_up_ds = create_npy(f'{mu_path}/4_{i}.npy', move_up_ds)

    for i in range(2, 21):
        move_down_ds = create_npy(f'{md_path}/4_{i}.npy', move_down_ds)

    for i in range(2, 11):
        no_movement_ds = create_npy(f'{nm_path}/4_{i}.npy', no_movement_ds)

    for i in range(1, 21):
        move_up_ds = create_npy(f'{mu_path}/4_{i}.npy', move_up_ds)

    for i in range(1, 21):
        move_down_ds = create_npy(f'{md_path}/4_{i}.npy', move_down_ds)

    for i in range(1, 11):
        no_movement_ds = create_npy(f'{nm_path}/4_{i}.npy', no_movement_ds)

    # save datasets
    np.save(f'{base_path}/move_down.npy', move_down_ds)
    np.save(f'{base_path}/move_up.npy', move_up_ds)
    np.save(f'{base_path}/no_movement.npy', no_movement_ds)
