import numpy as np

def reduce_seq(mfcc_data, reduce_factor):
    new_mfcc = []
    for x in range(len(mfcc_data)):
        new_data = np.zeros((mfcc_data[x].shape[0] // reduce_factor) + 1, mfcc_data[x].shape[1])
        new_data[0] = mfcc_data[x][0,:]
        c = 1
        idx = 1
        while x + reduce_factor < mfcc_data[x].shape[0]:
            new_data[c] = np.mean(mfcc_data[x][idx:idx+10,:], axis=0)
            c += 1
            idx += reduce_factor

        new_mfcc.append(new_data)
        new_mfcc = np.array(new_mfcc)

    np.save('mfcc_reduce'+str(reduce_factor)+'.npy', new_mfcc)

