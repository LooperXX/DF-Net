import os
import argparse
from tqdm import tqdm

PAD_token = 1
SOS_token = 3
EOS_token = 2
UNK_token = 0

parser = argparse.ArgumentParser(description='DF-Net')

parser.add_argument('-ds', '--dataset', help='dataset, kvr or woz', required=False, default='kvr')
parser.add_argument('-e', '--epoch', help='epoch num', required=False, type=int, default=1000)
parser.add_argument('-fixed', '--fixed', help='fix seeds', required=False, default=False)
parser.add_argument('-random_seed', '--random_seed', help='random_seed', required=False, default=1234)
parser.add_argument('-em_dim', '--embeddings_dim', help='word embeddings dim', type=int, required=False, default=128)
parser.add_argument('-hdd', '--hidden', help='Hidden size', required=False, default=128)
parser.add_argument('-bsz', '--batch', help='Batch_size', type=int, required=False, default=16)
parser.add_argument('-lr', '--learn', help='Learning Rate', required=False, default=0.001)
parser.add_argument('-dr', '--drop', help='Drop Out', required=False, default=0.2)
parser.add_argument('-um', '--unk_mask', help='mask out input token to UNK', type=int, required=False, default=1)
parser.add_argument('-gpu', '--gpu', help='use gpu', required=False, default=False)
parser.add_argument('-l', '--layer', help='Layer Number', required=False, default=3)
parser.add_argument('-l_r', '--layer_r', help='RNN Layer Number', required=False, default=2)
parser.add_argument('-lm', '--limit', help='Word Limit', required=False, default=-10000)
parser.add_argument('-path', '--path', help='path of the file to load', required=False)
parser.add_argument('-clip', '--clip', help='gradient clipping', required=False, default=10)
parser.add_argument('-count', '--count', help='count for early stop', required=False, type=int, default=8)
parser.add_argument('-tfr', '--teacher_forcing_ratio', help='teacher_forcing_ratio', type=float, required=False,
                    default=0.9)

parser.add_argument('-evalp', '--evalp', help='evaluation period', required=False, default=1)
parser.add_argument('-an', '--addName', help='An add name for the save folder', required=False, default='')
parser.add_argument('-gs', '--genSample', help='Generate Sample', required=False, default=0)
parser.add_argument('-es', '--earlyStop', help='Early Stop Criteria, BLEU or ENTF1', required=False, default='ENTF1')
parser.add_argument('-rec', '--record', help='use record function during inference', type=int, required=False,
                    default=1)
parser.add_argument('-op', '--output', help='output file', required=False, default='output.log')

args = vars(parser.parse_args())
print(str(args))
USE_CUDA = args['gpu']
print("USE_CUDA: " + str(USE_CUDA))

LIMIT = int(args["limit"])
MEM_TOKEN_SIZE = 6 if args["dataset"] == 'kvr' else 12
