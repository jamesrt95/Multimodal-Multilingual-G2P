import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import random
import math
import time


# define object to hold intermediate decoder states, sequences, and probabilities for beam search
class Beam(object):
    def __init__(self, hiddens, cells, att, word_list, logp, seq_len):
        super(Beam, self).__init__()
        self.hidden = hiddens
        self.cell = cells
        self.context = att
        self.word_list = word_list
        self.logp = logp
        self.logp_adj = logp # log probabilities after length normalization
        self.len = seq_len


# encoder for G2P model
class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_hid_dim, nlayers, dropout=0.0):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.nlayers = nlayers
        self.enc_hid_dim = enc_hid_dim
        self.rnns = nn.LSTM(self.embedding_dim, self.enc_hid_dim, self.nlayers, bidirectional=True, batch_first=True, dropout=dropout)

    def forward(self, data, seq_lens):
        # data: batch, seq, features (already padded)
        # seq_lens: list of lengths for each sequence in the batch
        output = self.embedding(data)
        output = rnn.pack_padded_sequence(output, seq_lens, batch_first=True)
        output, hidden = self.rnns(output)
        output, _ = rnn.pad_packed_sequence(output, batch_first=True)

        return output


# dot product attention
class Attention(nn.Module):
    def __init__(self, encode_dim, decode_dim, att_dim, device):
        super(Attention, self).__init__()
        self.att_dim = att_dim
        self.dec_proj = nn.Linear(decode_dim, att_dim)
        self.enc_proj = nn.Linear(encode_dim, att_dim)
        self.value_proj = nn.Linear(encode_dim, att_dim)
        self.device = device

    # q: N * decoder_hidden
    # v: N * source_seq * encoder_out
    def forward(self, q, v, input_seq_lens):
        batch_size = q.size(0)
        hidden_size = q.size(1)
        S = np.arange(v.size(1)).reshape(1, -1)
        L = np.array(input_seq_lens).reshape(-1, 1)
        # attention mask
        att_mask = S > L
        att_mask = torch.from_numpy(att_mask.astype(int)).type(torch.ByteTensor).to(self.device)
        # N * att_dim * 1
        speller_proj = self.dec_proj(q).unsqueeze(2)
        # N * source_seq * att_dim
        listener_proj = self.enc_proj(v)
        value = self.value_proj(v)

        # e will be N * source_seq * 1, so remove last dim to get N * source_seq
        e = torch.bmm(listener_proj, speller_proj).squeeze(dim=2)
        e = F.softmax(e.masked_fill_(att_mask, -1e10), dim=1)
        # N * 1 * source_seq x N * source_seq * encoder_out -> N * 1 * encoder_out
        c = torch.bmm(e.unsqueeze(1), value).squeeze(dim=1)
        return c


# decoder for text to speech
class SpeechDecoder(nn.Module):
    def __init__(self, feature_size, hidden_dim, nlayers, enc_out_dim, attention, device):
        super(SpeechDecoder, self).__init__()
        self.feature_size = feature_size
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers
        self.device = device
        self.attention = attention
        self.rnns = [nn.LSTMCell(self.feature_size + self.attention.att_dim, self.hidden_dim)]
        for x in range(1, self.nlayers):
            self.rnns.append(nn.LSTMCell(self.hidden_dim, self.hidden_dim))
        self.rnns = nn.ModuleList(self.rnns)

        self.fc = nn.Linear(self.attention.att_dim, self.feature_size)

    # targets: N * seq size padded tensor of target labels (seq includes sos/eos tokens)
    # encoder_out: N * seq * encoder_hidden size tensor from encoder
    def forward(self, padded_targets, encoder_out, input_seq_lens):
        batch_size = encoder_out.size(0)
        # initialize list of h, c states for the rnn cell layers
        h = [encoder_out.new_zeros((batch_size, self.hidden_dim))]
        c = [encoder_out.new_zeros((batch_size, self.hidden_dim))]
        for x in range(1, self.nlayers):
            h.append(encoder_out.new_zeros((batch_size, self.hidden_dim)))
            c.append(encoder_out.new_zeros((batch_size, self.hidden_dim)))
        attention_context = encoder_out.new_zeros(batch_size, self.attention.att_dim)

        predictions = []
        teach = 1.0
        for t in range(padded_targets.size(1) - 1):
            if t == 0 or random.random() < teach:
                prev_mfcc = padded_targets[:,t]
            else:
                prev_mfcc = pred
            rnn_input = torch.cat((prev_mfcc, attention_context), dim=1)
            h[0], c[0] = self.rnns[0](rnn_input, (h[0], c[0]))
            for x in range(1, self.nlayers):
                h[x], c[x] = self.rnns[x](h[x-1], (h[x], c[x]))

            # take the output from the final rnn layer
            output = h[-1]
            attention_context = self.attention(output, encoder_out, input_seq_lens)

            pred = self.fc(attention_context)
            predictions.append(pred)

        retval = torch.stack(predictions, dim=1)
        return retval


# decoder for text to IPA
class IPADecoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, nlayers, enc_out_dim, attention, device):
        super(IPADecoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim=embedding_dim
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.hidden_dim = hidden_dim
        self.nlayers = nlayers
        self.device = device
        self.attention = attention
        self.rnns = [nn.LSTMCell(self.embedding_dim + self.attention.att_dim, self.hidden_dim)]
        for x in range(1, self.nlayers):
            self.rnns.append(nn.LSTMCell(self.hidden_dim, self.hidden_dim))
        self.rnns = nn.ModuleList(self.rnns)

        self.fc = nn.Linear(self.attention.att_dim, vocab_size - 1)

    # targets: N * seq size padded tensor of target labels (seq includes sos/eos tokens)
    # encoder_out: N * seq * encoder_hidden size tensor from encoder
    def forward(self, padded_targets, encoder_out, input_seq_lens, teach=0.9):
        batch_size = encoder_out.size(0)
        # initialize list of h, c states for the rnn cell layers
        h = [encoder_out.new_zeros((batch_size, self.hidden_dim))]
        c = [encoder_out.new_zeros((batch_size, self.hidden_dim))]
        for x in range(1, self.nlayers):
            h.append(encoder_out.new_zeros((batch_size, self.hidden_dim)))
            c.append(encoder_out.new_zeros((batch_size, self.hidden_dim)))
        attention_context = encoder_out.new_zeros(batch_size, self.attention.att_dim)

        predictions = []
        for t in range(padded_targets.size(1) - 1):
            if t == 0 or random.random() < teach:
                emb = self.embedding(padded_targets[:,t])
            else:
                emb = self.embedding(Categorical(F.softmax(pred,dim=-1)).sample())
            rnn_input = torch.cat((emb, attention_context), dim=1)
            h[0], c[0] = self.rnns[0](rnn_input, (h[0], c[0]))
            for x in range(1, self.nlayers):
                h[x], c[x] = self.rnns[x](h[x-1], (h[x], c[x]))

            # take the output from the final rnn layer
            output = h[-1]
            attention_context = self.attention(output, encoder_out, input_seq_lens)

            pred = self.fc(attention_context)
            predictions.append(pred)

        retval = torch.stack(predictions, dim=1)
        return retval


    # greedy decoder (using sampling) to generate sequence
    def generate(self, encoder_out, input_seq_lens, token_data):
        batch_size = 1
        max_seq = 200
        EOS = token_data['ipa_to_idx']['EOS']
        # initialize list of h, c states for the rnn cell layers
        h = [encoder_out.new_zeros((batch_size, self.hidden_dim))]
        c = [encoder_out.new_zeros((batch_size, self.hidden_dim))]
        for x in range(1, self.nlayers):
            h.append(encoder_out.new_zeros((batch_size, self.hidden_dim)))
            c.append(encoder_out.new_zeros((batch_size, self.hidden_dim)))
        attention_context = encoder_out.new_zeros(batch_size, self.attention.att_dim)

        predictions = []
        for t in range(max_seq):
            if t == 0:
                sos = torch.tensor([EOS] * batch_size).to(self.device)
                emb = self.embedding(sos)
            else:
                emb = self.embedding(pred)
            rnn_input = torch.cat((emb, attention_context), dim=1)
            h[0], c[0] = self.rnns[0](rnn_input, (h[0], c[0]))
            for x in range(1, self.nlayers):
                h[x], c[x] = self.rnns[x](h[x-1], (h[x], c[x]))

            # take the output from the final rnn layer
            output = h[-1]
            attention_context = self.attention(output, encoder_out, input_seq_lens)

            dist = F.softmax(self.fc(attention_context),dim=-1)
            pred = Categorical(dist).sample()
            predictions.append(pred)
            if pred == EOS:
                break

        retval = torch.stack(predictions, dim=1)
        return retval


    # beam search generation
    def generate_beam(self, encoder_out, input_seq_lens, token_data, beam_size):
        batch_size = 1
        beam_size = beam_size
        max_seq = 200
        alpha = 0.65
        EOS = token_data['ipa_to_idx']['EOS']
        # initialize list of h, c states for the rnn cell layers
        h = [encoder_out.new_zeros((batch_size, self.hidden_dim))]
        c = [encoder_out.new_zeros((batch_size, self.hidden_dim))]
        for x in range(1, self.nlayers):
            h.append(encoder_out.new_zeros((batch_size, self.hidden_dim)))
            c.append(encoder_out.new_zeros((batch_size, self.hidden_dim)))
        attention_context = encoder_out.new_zeros(batch_size, self.attention.att_dim)

        beams = []
        predictions = []
        for t in range(max_seq):
            # special handling for first time step
            if t == 0:
                sos = torch.tensor([EOS]).to(self.device)
                emb = self.embedding(sos)
                rnn_input = torch.cat((emb, attention_context), dim=1)
                h[0], c[0] = self.rnns[0](rnn_input, (h[0], c[0]))
                for x in range(1, self.nlayers):
                    h[x], c[x] = self.rnns[x](h[x-1], (h[x], c[x]))

                # take the output from the final rnn layer
                attention_context = self.attention(h[-1], encoder_out, input_seq_lens)

                dist = F.softmax(self.fc(attention_context),dim=-1)
                idxes = torch.argsort(dist, descending=True).squeeze(0)
                for x in range(beam_size):
                    if len(beams) < beam_size:
                        beams.append(Beam([hidden.clone() for hidden in h], [cell.clone() for cell in c], attention_context.clone(), [idxes[x].item()], math.log(dist[0, idxes[x]].item()), 1))


            # build list of candidate beams and select the top k
            else:
                candidates = []
                for beam in beams:
                    if beam.word_list[-1] == EOS:
                        continue

                    next_char = torch.tensor([beam.word_list[-1]]).to(self.device)
                    emb = self.embedding(next_char)
                    rnn_input = torch.cat((emb, beam.context), dim=1)
                    h[0], c[0] = self.rnns[0](rnn_input, (beam.hidden[0], beam.cell[0]))
                    for x in range(1, self.nlayers):
                        h[x], c[x] = self.rnns[x](h[x-1], (beam.hidden[x], beam.cell[x]))

                    attention_context = self.attention(h[-1], encoder_out, input_seq_lens)
                    dist = F.softmax(self.fc(attention_context),dim=-1)
                    idxes = torch.argsort(dist, descending=True).squeeze(0)
                    for x in range(beam_size):
                        candidates.append(Beam([hidden.clone() for hidden in h], [cell.clone() for cell in c], attention_context.clone(), beam.word_list + [idxes[x].item()], beam.logp + math.log(dist[0, idxes[x]].item()), beam.len + 1))
                        # apply length normalization (using formula from Google's NMT paper)
                        candidates[-1].logp_adj = candidates[-1].logp / (((5 + candidates[-1].len) / 6)**alpha)

                candidates.sort(key=lambda x: x.logp, reverse=True)
                if len(candidates) > 0:
                    for y in range(beam_size):
                        if len(beams) < beam_size:
                            beams.append(candidates.pop(0))
                        elif beams[y].word_list[-1] == EOS:
                            try:
                                if beams[y].logp_adj < candidates[0].logp_adj:
                                    beams[y] = candidates.pop(0)
                            except:
                                import pdb; pdb.set_trace()
                        else:
                            beams[y] = candidates.pop(0)

                beams.sort(key=lambda x: x.logp_adj, reverse=True)

        return beams[0].word_list


# G2P model using encoder, attention, and decoder classes
class G2P(nn.Module):
    def __init__(self, grapheme_vocab_size, grapheme_emb_dim, enc_hid_dim, enc_layers, speech_hid_dim, speech_att_dim, mfcc_size, speech_layers, ipa_hid_dim, ipa_att_dim, ipa_vocab, ipa_emb_dim, ipa_layers, device):
        super(G2P, self).__init__()
        self.encoder = Encoder(grapheme_vocab_size, grapheme_emb_dim, enc_hid_dim, enc_layers)
        self.speech_attention = Attention(enc_hid_dim*2, speech_hid_dim, speech_att_dim, device)
        self.speech_decoder = SpeechDecoder(mfcc_size, speech_hid_dim, speech_layers, enc_hid_dim*2, self.speech_attention, device)
        self.ipa_attention = Attention(enc_hid_dim*2, ipa_hid_dim, ipa_att_dim, device)
        self.ipa_decoder = IPADecoder(ipa_vocab, ipa_emb_dim, ipa_hid_dim, ipa_layers, enc_hid_dim*2, self.ipa_attention, device)

    def forward(self, graphemes, input_seq_lens, mfcc_targets, ipa_targets, teach=0.9):
        enc_output = self.encoder(graphemes, input_seq_lens)
        mfcc_out = self.speech_decoder(mfcc_targets, enc_output, input_seq_lens)
        ipa_out = self.ipa_decoder(ipa_targets, enc_output, input_seq_lens, teach)
        return mfcc_out, ipa_out

    def generate(self, graphemes, input_seq_lens, token_data):
        enc_output = self.encoder(graphemes, input_seq_lens)
        preds = self.ipa_decoder.generate(enc_output, input_seq_lens, token_data)
        return preds

    def generate_beam(self, graphemes, input_seq_lens, token_data, beam_size=10):
        enc_output = self.encoder(graphemes, input_seq_lens)
        preds = self.ipa_decoder.generate_beam(enc_output, input_seq_lens, token_data, beam_size)
        return preds

