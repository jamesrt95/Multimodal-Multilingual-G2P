import numpy as np
import sys
import ujson as json


# build list of all grapheme characters and all IPA tokens in the trianing set
# also create dictionaries to convert charactters/tokens to list indices
# output: token_list.json, contains:
#   graphemes: list of all characters that occur in the grapheme data
#   grapheme_to_idx: dict to convert a grapheme character to index for the above list
#   ipa: list of all ipa tokens in phoneme data
#   ipa_to_idx dict to convert an ipa token to index for above list
def build_vocabulary():
    grapheme_chars = {}
    grapheme_char_list = []
    ipa = {}
    ipa_token_list = []
    with open("wilderness_graphemes.txt") as g_data, open("wilderness_phonemes.txt") as p_data:
        for x in range(10000):
            g_line = g_data.readline().strip()
            for char in g_line[4:]:
                if char not in grapheme_chars:
                    grapheme_char_list.append(char)
                    grapheme_chars[char] = len(grapheme_char_list) - 1

            p_line = p_data.readline().strip()
            ipa_tokens = p_line.split()
            for token in ipa_tokens:
                if token not in ipa:
                    ipa_token_list.append(token)
                    ipa[token] = len(ipa_token_list) - 1

    with open("wilderness_dev_graphemes.txt") as g_data, open("wilderness_dev_phonemes.txt") as p_data:
            for x in range(10000):
                g_line = g_data.readline().strip()
                for char in g_line[4:]:
                    if char not in grapheme_chars:
                        grapheme_char_list.append(char)
                        grapheme_chars[char] = len(grapheme_char_list) - 1

                p_line = p_data.readline().strip()
                ipa_tokens = p_line.split()
                for token in ipa_tokens:
                    if token not in ipa:
                        ipa_token_list.append(token)
                        ipa[token] = len(ipa_token_list) - 1

    # add EOS and padding chars/tokens for graphemes and phonemes
    grapheme_char_list.append('EOS')
    grapheme_chars['EOS'] = len(grapheme_char_list) - 1
    grapheme_char_list.append('PAD')
    grapheme_chars['PAD'] = len(grapheme_char_list) - 1
    ipa_token_list.append('EOS')
    ipa['EOS'] = len(ipa_token_list) - 1
    ipa_token_list.append('PAD')
    ipa['PAD'] = len(ipa_token_list) - 1

    json_out = {}
    json_out["graphemes"] = grapheme_char_list
    json_out["grapheme_to_idx"] = grapheme_chars
    json_out["ipa"] = ipa_token_list
    json_out["ipa_to_idx"] = ipa
    for x in range(len(grapheme_char_list)):
        assert grapheme_chars[grapheme_char_list[x]] == x
    for x in range(len(ipa_token_list)):
        assert ipa[ipa_token_list[x]] == x
    with open("token_list.json", "w") as outfile:
        json.dump(json_out, outfile)


# read in grapheme, phoneme, and mfcc data and build npy arrays
# write out npy files for graphemes, phonemes, mfcc data, and language tags
# each element in the npy file corresponds to the same training data instance
# output:
#   graphemes.npy: numpy array, each element is a numpy array of grapheme indices for one training instance
#   langs.npy: numpy array containing language tag for each training instance
#   phonemes.npy: numpy array, each element is a numpy array of ipa tokens for one training instance
#   wilderness_mfcc.npy: numpy array, each element is a 2D numpy array of mfcc coefficients for one training instance
def build_dataset(grapheme_file, phoneme_file, wav_file, output_dir):
    with open("token_list.json") as dict_files:
        g2p_dicts = json.load(dict_files)

    # get the grapheme data and write to file
    grapheme_data = []
    lang_tags = []
    with open(grapheme_file) as g_data:
        for g_line in g_data:
            g_line = g_line.strip()
            lang_tag = g_line[:3]
            grapheme_text = g_line[4:]
            grapheme_idxs = np.array([g2p_dicts["grapheme_to_idx"][char] if char in g2p_dicts["grapheme_to_idx"] else g2p_dicts["grapheme_to_idx"][' '] for char in grapheme_text])
            grapheme_data.append(grapheme_idxs)
            lang_tags.append(lang_tag)

        grapheme_np = np.array(grapheme_data)
        lang_np = np.array(lang_tags)
        np.save(output_dir + '/' + 'graphemes.npy', grapheme_np)
        np.save(output_dir + '/' + 'langs.npy', lang_np)

    # get the phoneme data and write to file
    phoneme_data = []
    with open(phoneme_file) as p_data:
        for p_line in p_data:
            p_line = p_line.strip().split()
            phoneme_idxs = np.array([g2p_dicts["ipa_to_idx"][token] if token in g2p_dicts["ipa_to_idx"] for token in p_line])
            phoneme_data.append(phoneme_idxs)

    phoneme_np = np.array(phoneme_data)
    np.save(output_dir + '/' + 'phonemes.npy', phoneme_np)

    # get the mfcc data and write to file
    mfcc_data = []
    missing_files = []
    with open(wav_file) as w_data:
        for w_line in w_data:
            w_line = w_line.strip()

            w_file, w_graph = w_line.split('|')
            lang_code = w_graph.split()[0]
            f_name = w_file.split('/')[1]
            f_name = f_name[:f_name.find('.')]
            # if an mfcc file is missing, record it and dump all filenames later
            try:
                mfcc = np.load('mfcc_npy/' + lang_code + '/' + f_name + '.npy', encoding='bytes')
            except:
                missing_files.append('mfcc_npy/' + lang_code + '/' + f_name + '.npy')
                continue
            mfcc_data.append(mfcc)

    with open(output_dir + '/' + 'missing_files.json', 'w') as outfile:
        for name in missing_files:
            outfile.write(name + '\n')

    mfcc_np = np.array(mfcc_data)
    np.save(output_dir + '/' + 'wilderness_mfcc.npy', mfcc_np)


def main():
    grapheme_file = sys.argv[1]
    phoneme_file = sys.argv[2]
    wav_file = sys.argv[3]
    output_dir = sys.argv[4]
    build_vocabulary()
    build_dataset(grapheme_file, phoneme_file, wav_file, output_dir)


if __name__ == "__main__":
    main()
