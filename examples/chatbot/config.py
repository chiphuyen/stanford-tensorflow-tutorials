# dataset parameters

DATA_PATH = '/Users/Chip/data/cornell movie-dialogs corpus'
CONVO_FILE = 'movie_conversations.txt'
LINE_FILE = 'movie_lines.txt'
PROCESSED_PATH = 'processed'
CPT_PATH = 'checkpoints'

ENC_VOCAB = 20000
DEC_VOCAB = 20000

UNK_ID = ENC_VOCAB
START_ID = ENC_VOCAB + 1
END_ID = ENC_VOCAB + 2
PAD_ID = ENC_VOCAB + 3

TESTSET_SIZE = 30000

# model parameters
""" Encode train length distribution:
[175, 92, 11883, 8387, 10656, 13613, 13480, 12850, 11802, 10165, 
8973, 7731, 7005, 6073, 5521, 5020, 4530, 4421, 3746, 3474, 3192, 
2724, 2587, 2413, 2252, 2015, 1816, 1728, 1555, 1392, 1327, 1248, 
1128, 1084, 1010, 884, 843, 755, 705, 660, 649, 594, 558, 517, 475, 
426, 444, 388, 349, 337]
These buckets size seem to work the best
"""
# BUCKETS = [(7, 9), (9, 11), (11, 13), (14, 16), (17, 20), (20, 23), (24, 27), (30, 33), (40, 45)]
BUCKETS = [(6, 8), (8, 10), (10, 12), (13, 15), (16, 19), (19, 22), (23, 26), (29, 32), (39, 44)]

NUM_LAYERS = 3
HIDDEN_SIZE = 256
BATCH_SIZE = 64

LR = 0.5
LR_DECAY_FACTOR = 0.99
MAX_GRAD_NORM = 5.0

SKIP_STEP = 100
NUM_SAMPLES = 512