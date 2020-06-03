import torch
from torch.autograd import Variable
from math import ceil
from torch.utils.data import TensorDataset
from torchtext.datasets.text_classification import _create_data_from_iterator, _csv_iterator, TextClassificationDataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab import Vocab
from torchtext.utils import download_from_url, extract_archive, unicode_csv_reader
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
import numpy as np


def prepare_generator_batch(samples, start_letter=0, gpu=False):
    """
    Takes samples (a batch) and returns

    Inputs: samples, start_letter, cuda
        - samples: batch_size x seq_len (Tensor with a sample in each row)

    Returns: inp, target
        - inp: batch_size x seq_len (same as target, but with start_letter prepended)
        - target: batch_size x seq_len (Variable same as samples)
    """

    batch_size, seq_len = samples.size()

    inp = torch.zeros(batch_size, seq_len)
    target = samples
    inp[:, 0] = start_letter
    inp[:, 1:] = target[:, :seq_len-1]

    inp = Variable(inp).type(torch.LongTensor)
    target = Variable(target).type(torch.LongTensor)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


def prepare_discriminator_data(pos_samples, neg_samples, gpu=False):
    """
    Takes positive (target) samples, negative (generator) samples and prepares inp and target data for discriminator.

    Inputs: pos_samples, neg_samples
        - pos_samples: pos_size x seq_len
        - neg_samples: neg_size x seq_len

    Returns: inp, target
        - inp: (pos_size + neg_size) x seq_len
        - target: pos_size + neg_size (boolean 1/0)
    """
    inp = torch.cat((pos_samples, neg_samples), 0).type(torch.LongTensor)
    target = torch.ones(pos_samples.size()[0] + neg_samples.size()[0])
    target[pos_samples.size()[0]:] = 0

    # shuffle
    perm = torch.randperm(target.size()[0])
    target = target[perm]
    inp = inp[perm]

    inp = Variable(inp)
    target = Variable(target)

    if gpu:
        inp = inp.cuda()
        target = target.cuda()

    return inp, target


def batchwise_sample(gen, num_samples, batch_size):
    """
    Sample num_samples samples batch_size samples at a time from gen.
    Does not require gpu since gen.sample() takes care of that.
    """

    samples = []
    for i in range(int(ceil(num_samples/float(batch_size)))):
        samples.append(gen.sample(batch_size))

    return torch.cat(samples, 0)[:num_samples]


def batchwise_oracle_nll(gen, oracle, num_samples, batch_size, max_seq_len, start_letter=0, gpu=False):
    s = batchwise_sample(gen, num_samples, batch_size)
    oracle_nll = 0
    for i in range(0, num_samples, batch_size):
        inp, target = prepare_generator_batch(s[i:i+batch_size], start_letter, gpu)
        oracle_loss = oracle.batchNLLLoss(inp, target) / max_seq_len
        oracle_nll += oracle_loss.data.item()

    return oracle_nll/(num_samples/batch_size)

def _setup_datasets(dataset_name, root='.data', ngrams=1, vocab=None, include_unk=False):
    #dataset_tar = download_from_url(URLS[dataset_name], root=root)
    extracted_files = extract_archive('.data/ag_news_csv.tar.gz')

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname

    if vocab is None:
        print('Building Vocab based on {}'.format(train_csv_path))
        vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))
    else:
        if not isinstance(vocab, Vocab):
            raise TypeError("Passed vocabulary is not of type Vocab")
    print('Vocab has {} entries'.format(len(vocab)))
    print('Creating training data')
    train_data, train_labels = _create_data_from_iterator(
        vocab, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
    print('Creating testing data')
    test_data, test_labels = _create_data_from_iterator(
        vocab, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (TextClassificationDataset(vocab, train_data, train_labels),
            TextClassificationDataset(vocab, test_data, test_labels), vocab)


def init_dataset(labeled_num):
    from main import CLASS_NUM
    raw_dataset, test_dataset, vocab = _setup_datasets('.data')
    class_tot = [0] * CLASS_NUM
    label_data = []
    labels = []
    tot = 0
    perm = np.random.permutation(raw_dataset.__len__())
    unlabel = []
    for i in range(raw_dataset.__len__()):
        label, datum = raw_dataset.__getitem__(perm[i])
        d = datum.numpy()
        if len(d) < 20:
            d = np.pad(d, (0, 20-len(d)), 'constant', constant_values=(0, 1))
        elif len(d) > 20:
            d = d[:20]
        if class_tot[label] < labeled_num:
            label_data.append(d)
            labels.append(label)
            class_tot[label] += 1
            tot += 1
        if tot >= CLASS_NUM * labeled_num and len(unlabel) <= 10000:
            unlabel.append(d)
    fake_label = np.zeros(len(unlabel))
    test_data, test_label = [], []
    for i in range(test_dataset.__len__()):
        label, datum = raw_dataset.__getitem__(perm[i])
        d = datum.numpy()
        if len(d) < 210:
            d = np.pad(d, (0, 210-len(d)), 'constant', constant_values=(0, 1))
        test_data.append(d)
        test_label.append(label)

    return TensorDataset(torch.IntTensor(np.array(label_data)), torch.LongTensor(np.array(labels))), \
           TensorDataset(torch.IntTensor(np.array(unlabel)), torch.LongTensor(np.array(fake_label))), \
           TensorDataset(torch.IntTensor(np.array(test_data)), torch.LongTensor(np.array(test_label))), vocab