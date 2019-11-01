import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sys
import ujson as json

class LoadData(Dataset):

    def __init__(self, g_file, mfcc_file, p_file, lang_file, norms_file, test=False):
        self.graphemes = np.load(g_file, allow_pickle=True)
        self.test = test
        self.lang_tags = np.load(lang_file, allow_pickle=True)
        if not self.test:
            self.phonemes = np.load(p_file, allow_pickle=True)
            self.mfcc = np.load(mfcc_file, allow_pickle=True)
            self.norms = np.load(norms_file, allow_pickle=True)
            for x in range(self.mfcc.shape[0]):
                self.mfcc[x] = self.norms[2] * (self.mfcc[x] - self.norms[1]) / (self.norms[0] - self.norms[1])

        else:
            self.mfcc = None
        with open('small_training/token_list.json') as idx_data:
            self.g2p_idx = json.load(idx_data)


    def __getitem__(self, index):
        if self.test:
            return torch.tensor(self.graphemes[index]), None, None, self.lang_tags[index]
        eos = [self.g2p_idx['ipa_to_idx']['EOS']]
        phonemes = np.concatenate((eos, self.phonemes[index], eos))
        return torch.tensor(self.graphemes[index]), torch.tensor(self.mfcc[index]).float(), torch.tensor(phonemes), self.lang_tags[index]

    def __len__(self):
        return self.graphemes.shape[0]

# return batch of (graphemes, mfcc data, phonemes, language tags, sequence_order)
# CAUTION: all data are sorted in descending order by grapheme sequence length 
# return values:
#   graphemes: batch_size list of tensors containing grapheme character indices
#   mfcc: batch_size list of sequence_length * feature_dim tensors containing mfcc coefficients
#   phonemes: batch_size list of tensors containing phoneme tokens
#   language tags: batch_size numpy array of strings containing the language tag for each training instance
#   sequence_order: indices for the permutation used to sort the data by length (needed to recover original order)
def collate(seq_list):
    graphemes, mfcc, phonemes, langs = zip(*seq_list)
    g_lens = [seq.shape[0] for seq in graphemes]
    seq_order = sorted(range(len(g_lens)), key=g_lens.__getitem__, reverse=True)
    graphemes = [graphemes[i] for i in seq_order]
    if mfcc:
        mfcc = [mfcc[i] for i in seq_order]
    if phonemes:
        phonemes = [phonemes[i] for i in seq_order]
    langs = [langs[i] for i in seq_order]
    return graphemes, mfcc, phonemes, langs, seq_order

