import torch
import torch.utils.data as data
import torch.nn as nn
from utils.config import *


def _cuda(x):
    if USE_CUDA:
        return x.cuda()
    else:
        return x


if args['dataset'] == 'kvr':
    domains = {'navigate': 0, 'weather': 1, 'schedule': 2}
elif args['dataset'] == 'woz':
    domains = {'restaurant': 0, 'attraction': 1, 'hotel': 2}


class Lang:
    def __init__(self):
        self.word2index = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: 'UNK'}
        self.n_words = len(self.index2word)  # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])

        self.char2index = {}
        self.index2char = {PAD_token: "PAD", UNK_token: 'UNK'}
        self.n_chars = len(self.index2char)
        self.char2index = dict([(v, k) for k, v in self.index2char.items()])

    def index_words(self, story, trg=False):
        if trg:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1
        for char in word:
            if char not in self.char2index:
                self.char2index[char] = self.n_chars
                self.index2char[self.n_chars] = char
                self.n_chars += 1


class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""

    def __init__(self, data_info, src_word2id, trg_word2id, lang):
        """Reads source and target sequences from txt files."""
        self.data_info = {}
        for k in data_info.keys():
            self.data_info[k] = data_info[k]

        self.num_total_seqs = len(data_info['context_arr'])
        self.src_word2id = src_word2id
        self.trg_word2id = trg_word2id
        self.lang = lang

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        context_arr = self.data_info['context_arr'][index]
        context_arr = self.preprocess(context_arr, self.src_word2id, trg=False)
        response = self.data_info['response'][index]
        response = self.preprocess(response, self.trg_word2id)
        ptr_index = torch.Tensor(self.data_info['ptr_index'][index])
        selector_index = torch.Tensor(self.data_info['selector_index'][index])
        conv_arr = self.data_info['conv_arr'][index]
        conv_char_arr, conv_char_length = self.preprocess(conv_arr, self.src_word2id, trg=False, char=True)
        conv_arr = self.preprocess(conv_arr, self.src_word2id, trg=False)
        kb_arr = self.data_info['kb_arr'][index]
        kb_arr = self.preprocess(kb_arr, self.src_word2id, trg=False)
        sketch_response = self.data_info['sketch_response'][index]
        sketch_response = self.preprocess(sketch_response, self.trg_word2id)

        # processed information
        data_info = {}
        for k in self.data_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = self.data_info[k][index]

        data_info['conv_char_arr'] = conv_char_arr
        data_info['conv_char_length'] = conv_char_length

        # additional plain information
        data_info['context_arr_plain'] = self.data_info['context_arr'][index]
        data_info['response_plain'] = self.data_info['response'][index]
        data_info['gold_sketch_response'] = self.data_info['sketch_response'][index]
        data_info['kb_arr_plain'] = self.data_info['kb_arr'][index]

        return data_info

    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2id, trg=True, char=False):
        """Converts words to ids."""
        if trg:
            story = [word2id[word] if word in word2id else UNK_token for word in sequence.split(' ')] + [EOS_token]
        elif char:
            length = torch.Tensor([len(word[0]) for word in sequence])
            char_arr = []
            for word in sequence:
                for char in word[0]:
                    temp = self.lang.char2index[char] if char in self.lang.char2index else UNK_token
                    char_arr.append(temp)
            return torch.Tensor(char_arr), length
        else:
            story = []
            for i, word_triple in enumerate(sequence):
                story.append([])
                for ii, word in enumerate(word_triple):
                    temp = word2id[word] if word in word2id else UNK_token
                    story[i].append(temp)
        story = torch.Tensor(story)
        return story

    def collate_fn(self, data):
        def merge(sequences, story_dim):
            lengths = [len(seq) for seq in sequences]
            max_len = 1 if max(lengths) == 0 else max(lengths)
            if (story_dim):
                padded_seqs = torch.ones(len(sequences), max_len, MEM_TOKEN_SIZE).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    if len(seq) != 0:
                        padded_seqs[i, :end, :] = seq[:end]
            else:
                padded_seqs = torch.ones(len(sequences), max_len).long()
                for i, seq in enumerate(sequences):
                    end = lengths[i]
                    padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        def merge_index(sequences):
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).float()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs, lengths

        def merge_char(chars, length, max_seq_len):
            max_word_len = max([max(leng.long())] for leng in length)[0].item()
            seqs_char = torch.ones((len(length), max_seq_len, max_word_len))
            seqs_char_lenghth = torch.ones((len(length), max_seq_len))
            for i, leng in enumerate(length):
                seqs_char_lenghth[i, :len(leng)] = leng
                start = 0
                for ii, word_len in enumerate(leng.long()):
                    seqs_char[i][ii][:word_len] = chars[i][start:start + word_len]
                    start += word_len
            seqs_char = seqs_char.view(-1, max_word_len)
            seqs_char_lenghth = seqs_char_lenghth.view(seqs_char.size(0), )
            seqs_char_lenghth, char_perm_idx = seqs_char_lenghth.sort(0, descending=True)
            seqs_char = seqs_char[char_perm_idx]
            _, char_seq_recover = char_perm_idx.sort(0, descending=False)
            return seqs_char, seqs_char_lenghth, char_seq_recover

        # sort a list by sequence length (descending order) to use pack_padded_sequence
        data.sort(key=lambda x: len(x['conv_arr']), reverse=True)
        item_info = {}
        for key in data[0].keys():
            item_info[key] = [d[key] for d in data]

        # merge sequences 
        context_arr, context_arr_lengths = merge(item_info['context_arr'], True)
        response, response_lengths = merge(item_info['response'], False)
        selector_index, _ = merge_index(item_info['selector_index'])
        ptr_index, _ = merge(item_info['ptr_index'], False)
        conv_arr, conv_arr_lengths = merge(item_info['conv_arr'], True)
        sketch_response, _ = merge(item_info['sketch_response'], False)
        kb_arr, kb_arr_lengths = merge(item_info['kb_arr'], True)

        max_seq_len = conv_arr.size(1)
        conv_char_arr, conv_char_length, char_seq_recover = merge_char(item_info['conv_char_arr'],
                                                                       item_info['conv_char_length'], max_seq_len)
        label_arr = _cuda(torch.Tensor([domains[label] for label in item_info['domain']]).long().unsqueeze(-1))
        # convert to contiguous and cuda
        context_arr = _cuda(context_arr.contiguous())
        response = _cuda(response.contiguous())
        selector_index = _cuda(selector_index.contiguous())
        ptr_index = _cuda(ptr_index.contiguous())
        conv_arr = _cuda(conv_arr.transpose(0, 1).contiguous())
        sketch_response = _cuda(sketch_response.contiguous())
        conv_char_arr = _cuda(conv_char_arr.contiguous())
        conv_char_length = _cuda(conv_char_length.contiguous())
        char_seq_recover = _cuda(char_seq_recover.contiguous())

        if (len(list(kb_arr.size())) > 1): kb_arr = _cuda(kb_arr.transpose(0, 1).contiguous())

        item_info['conv_char_arr'] = 0
        item_info['conv_char_length'] = 0
        item_info['char_seq_recover'] = 0
        item_info['label_arr'] = 0

        # processed information
        data_info = {}
        for k in item_info.keys():
            try:
                data_info[k] = locals()[k]
            except:
                data_info[k] = item_info[k]

        # additional plain information
        data_info['context_arr_lengths'] = context_arr_lengths
        data_info['response_lengths'] = response_lengths
        data_info['conv_arr_lengths'] = conv_arr_lengths
        data_info['kb_arr_lengths'] = kb_arr_lengths
        return data_info


def get_seq(pairs, lang, batch_size, type):
    data_info = {}
    for k in pairs[0].keys():
        data_info[k] = []

    for pair in pairs:
        for k in pair.keys():
            data_info[k].append(pair[k])
        if (type):
            lang.index_words(pair['context_arr'])
            lang.index_words(pair['response'], trg=True)
            lang.index_words(pair['sketch_response'], trg=True)

    dataset = Dataset(data_info, lang.word2index, lang.word2index, lang)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=type,
                                              collate_fn=dataset.collate_fn,
                                              # num_workers=8,
                                              drop_last=True)
    return data_loader
