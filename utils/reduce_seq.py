import numpy as np
import sys

# read in mfcc data and average mffcs over a given number of frames
# this is to reduce the length of the data and make computations easier
def reduce_seq(source_file, reduce_factor):
    mfcc_data = np.load(source_file, allow_pickle=True)
    new_mfcc = []
    for x in range(len(mfcc_data)):
        new_data = np.zeros(((mfcc_data[x].shape[0] // reduce_factor) + 1, mfcc_data[x].shape[1]))
        new_data[0] = mfcc_data[x][0,:]
        idx = 1
        c = 1
        while (idx + reduce_factor) < mfcc_data[x].shape[0]:
            new_data[c] = np.mean(mfcc_data[x][idx:idx+reduce_factor,:], axis=0)
            c += 1
            idx += reduce_factor

        if np.sum(new_data[-1], axis=0) == 0:
            new_data = new_data[:-1]
        new_mfcc.append(new_data)

    new_mfcc = np.array(new_mfcc)
    np.save(source_file.split('.')[0] + str(reduce_factor) + '.npy', new_mfcc)


def main():
    source_file = sys.argv[1]
    reduce_factor = int(sys.argv[2])
    reduce_seq(source_file, reduce_factor)


if __name__ == "__main__":
    main()
