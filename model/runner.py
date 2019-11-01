import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import os
import sys
import ujson as json
from dataloader import LoadData, collate
from model import G2P
import resource


def train_epoch(model, train_loader, ipa_criterion, speech_criterion, optimizer, token_data, teach, loss_scale):
    model.train()
    model.to(device)
    ipa_criterion.to(device)
    speech_criterion.to(device)

    running_ipa_loss = 0.
    running_mfcc_loss = 0.
    batch_count = 0

    for graphemes, mfcc, phonemes, langs, seq_order in train_loader:
        batch_size = len(graphemes)
        optimizer.zero_grad()
        grapheme_lens = [seq.shape[0] for seq in graphemes]
        mfcc_lens = [seq.shape[0] for seq in mfcc]
        mfcc_lens_tensor = torch.Tensor(mfcc_lens).to(device)
        phoneme_lens = torch.Tensor([seq.shape[0] for seq in phonemes]).to(device)

        graphemes = rnn.pad_sequence(graphemes, batch_first=True, padding_value=token_data['grapheme_to_idx']['PAD']).to(device)
        phonemes = rnn.pad_sequence(phonemes, batch_first=True, padding_value=token_data['ipa_to_idx']['PAD']).to(device)
        mfcc = rnn.pad_sequence(mfcc, batch_first=True).to(device)
        mfcc_out, ipa_out = model(graphemes, grapheme_lens, mfcc, phonemes, teach)
        ipa_out = ipa_out.view(-1, ipa_out.size(-1))

        phonemes = phonemes[:,1:].contiguous().view(-1)
        loss1 = ipa_criterion(ipa_out, phonemes)
        ipa_loss = torch.sum(ipa_criterion(ipa_out, phonemes)) / torch.sum(phoneme_lens - 1)
        # mask the mfcc loss
        mfcc_out = mfcc_out.view(-1, mfcc_out.size(-1))
        S = np.arange(mfcc.size(1) - 1).reshape(1, -1)
        L = np.array(mfcc_lens).reshape(-1, 1) - 1
        mfcc_mask = torch.from_numpy((L > S) * 1).float().view(-1).unsqueeze(1).to(device)
        mfcc = mfcc[:,1:,:].contiguous().view(-1, mfcc.size(-1))
        mfcc_loss = speech_criterion(mfcc_out, mfcc)
        #mfcc_loss = torch.sum(mfcc_loss * mfcc_mask) / (torch.sum(mfcc_lens_tensor - 1) * mfcc.size(-1))
        mfcc_loss = torch.sum(mfcc_loss * mfcc_mask) / torch.sum(mfcc_lens_tensor - 1)
        running_ipa_loss += ipa_loss.item()
        running_mfcc_loss += mfcc_loss.item()
        loss = ipa_loss + loss_scale * mfcc_loss

        loss.backward()
        optimizer.step()
        batch_count += 1
        if batch_count % 20 == 0:
            print(batch_count, running_ipa_loss / batch_count, running_mfcc_loss / batch_count)

        del graphemes
        del phonemes
        del mfcc
        del mfcc_lens_tensor
        del phoneme_lens
        del loss
        del ipa_loss
        del mfcc_loss
        torch.cuda.empty_cache()

    running_ipa_loss /= batch_count
    running_mfcc_loss /= batch_count
    print("Training loss:", running_ipa_loss, running_mfcc_loss)
    return running_ipa_loss


def check_dev(model, dev_loader, ipa_criterion, speech_criterion, token_data, teach):
    model.eval()
    model.to(device)
    ipa_criterion.to(device)
    speech_criterion.to(device)

    running_ipa_loss = 0.
    running_mfcc_loss = 0.
    dev_loss = 0.
    batch_count = 0

    for graphemes, mfcc, phonemes, langs, seq_order in dev_loader:
        batch_size = len(graphemes)
        grapheme_lens = [seq.shape[0] for seq in graphemes]
        mfcc_lens = [seq.shape[0] for seq in mfcc]
        mfcc_lens_tensor = torch.Tensor(mfcc_lens).to(device)
        phoneme_lens = torch.Tensor([seq.shape[0] for seq in phonemes]).to(device)

        graphemes = rnn.pad_sequence(graphemes, batch_first=True, padding_value=token_data['grapheme_to_idx']['PAD']).to(device)
        phonemes = rnn.pad_sequence(phonemes, batch_first=True, padding_value=token_data['ipa_to_idx']['PAD']).to(device)
        mfcc = rnn.pad_sequence(mfcc, batch_first=True).to(device)
        mfcc_out, ipa_out = model(graphemes, grapheme_lens, mfcc, phonemes, teach)
        ipa_out = ipa_out.view(-1, ipa_out.size(-1))

        phonemes = phonemes[:,1:].contiguous().view(-1)
        ipa_loss = torch.sum(ipa_criterion(ipa_out, phonemes)) / torch.sum(phoneme_lens - 1)
        # mask the mfcc loss
        mfcc_out = mfcc_out.view(-1, mfcc_out.size(-1))
        S = np.arange(mfcc.size(1) - 1).reshape(1, -1)
        L = np.array(mfcc_lens).reshape(-1, 1) - 1
        mfcc_mask = torch.from_numpy((L > S) * 1).float().view(-1).unsqueeze(1).to(device)
        mfcc = mfcc[:,1:,:].contiguous().view(-1, mfcc.size(-1))
        mfcc_loss = speech_criterion(mfcc_out, mfcc)
        mfcc_loss = torch.sum(mfcc_loss * mfcc_mask) / (torch.sum(mfcc_lens_tensor - 1) * mfcc.size(-1))
        running_ipa_loss += ipa_loss.item()
        running_mfcc_loss += mfcc_loss.item()

        batch_count += 1
        del graphemes
        del phonemes
        del mfcc
        del mfcc_lens_tensor
        del phoneme_lens
        del ipa_loss
        del mfcc_loss
        torch.cuda.empty_cache()

    print("dev loss:", running_ipa_loss / batch_count, running_mfcc_loss / batch_count)
    return running_ipa_loss / batch_count


def eval_test(model, test_loader, token_data, outfile):
    model.eval()
    model.to(device)

    batch_count = 0
    outputs = []

    for graphemes, mfcc, phonemes, langs, seq_order in test_loader:
        batch_size = len(graphemes)
        grapheme_lens = [seq.shape[0] for seq in graphemes]

        graphemes = rnn.pad_sequence(graphemes, batch_first=True, padding_value=token_data['grapheme_to_idx']['PAD']).to(device)
        preds = model.generate_beam(graphemes, grapheme_lens, token_data)
        ipa_str = (langs[0] + ' ' + ' '.join([token_data['ipa'][x] for x in preds[:-1]])).strip()
        outputs.append(ipa_str.strip() + '\n')

        batch_count += 1
        del graphemes
        del phonemes
        del mfcc
        torch.cuda.empty_cache()

        if (batch_count + 1) % 50 == 0:
            print(batch_count + 1)

    with open(outfile, 'w') as ipa_out:
        for ipa_pred in outputs:
            ipa_out.write(ipa_pred)


if __name__ == "__main__":
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    train = True
    test = False
    #load_model = 'models_norm_mfcc/e12.mdl'
    #load_model = 'models_nomfcc/e10.mdl'
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
    MFCC_DIM = 39

    with open('small_training/token_list.json') as token_file:
        token_data = json.load(token_file)

    if train:
        train_set = LoadData("small_training/small_training_out/graphemes.npy", "small_training/small_training_out/wilderness_mfcc10.npy", "small_training/small_training_out//phonemes.npy", "small_training/small_training_out/langs.npy", "mfcc_norms.npy")
        train_generator = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8, collate_fn=collate)
        dev_set = LoadData("small_training/dev_set/graphemes.npy", "small_training/dev_set/wilderness_mfcc10.npy", "small_training/dev_set/phonemes.npy", "small_training/dev_set/langs.npy", "mfcc_norms.npy")
        dev_generator = DataLoader(dev_set, batch_size=10, shuffle=False, num_workers=8, collate_fn=collate)
        print("loaded", len(train_set), "training set inputs,", len(dev_set), "dev set inputs")
    if test:
        test_set = LoadData("test_sets/in_domain/graphemes.npy", None, None, "test_sets/in_domain/langs.npy", None, test=True)
        test_generator = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate)

    # model input parameters:
    # encoder vocab size
    # encoder embedding dim
    # encoder hidden dim
    # encoder layers
    # speech decoder hidden dim
    # speech decoder attention dim
    # mfcc features
    # speech decoder layers
    # ipa decoder hidden dim
    # ipa decoder attention dim
    # ipa vocab size
    # ipa decoder embedding size
    # ipa decoder layers
    model = G2P(len(token_data['graphemes']), 64, 128, 1, 128, 64, MFCC_DIM, 1, 128, 64, len(token_data['ipa']), 64, 1, device)
    #print(model)
    if load_model:
        model.load_state_dict(torch.load(load_model, map_location=device))
    ipa_criterion = nn.CrossEntropyLoss(ignore_index=token_data['ipa_to_idx']['PAD'], reduction='none')
    speech_criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if train:
        dev_losses = []
        teach = 0.9
        loss_scale = 0.1
        for x in range(50):
            print("starting epoch", x, "...")
            train_epoch(model, train_generator, ipa_criterion, speech_criterion, optimizer, token_data, teach, loss_scale)
            dl = check_dev(model, dev_generator, ipa_criterion, speech_criterion, token_data, teach)
            dev_losses.append(dl)
            torch.save(model.state_dict(), os.path.join('models_small_train', 'e{}.mdl'.format(x)))
            if x > 10:
                loss_scale *= 0.5
            if len(dev_losses) > 2:
                #if dev_losses[-3] - dev_losses[-2] < 0.01 and dev_losses[-2] - dev_losses[-1] < 0.01:
                if (x + 1) == 10 or (x + 1) == 15:
                    for param_group in optimizer.param_groups:
                        print('lowering learning rate', param_group['lr'], 'by factor of 10')
                        print(dev_losses[-1], dev_losses[-2])
                        param_group['lr'] *= 0.1

    if test:
        eval_test(model, test_generator, token_data, "preds_in_domain_nomfcc.txt")
