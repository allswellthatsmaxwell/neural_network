
import sys
sys.path.append('..')
import csv, string, re, math
import numpy as np
from sklearn.metrics import roc_auc_score
from prediction_utils import (trn_val_tst, 
                              standard_binary_classification_layers,
                              bind_and_sort)
import neural_network.neural_network as nn
import neural_network.activations as avs
import neural_network.loss_functions as losses
import matplotlib.pyplot as plt

def get_words(text):
    """Removes all punctuation from text, and collapses all whitespace
    characters to a single space"""
    return WS_COLLAPSE_RE.sub(' ', text.lower().translate(PUNCT_REMOVES))

def read_texts_stars(csv_path, maxrows = math.inf):
    """return the 'text' and 'stars' entries (in two lists) from the 
    first maxrows records in csv_path"""
    texts, stars, useful, funny, cool = [], [], [], [], []
    nrow = 0
    with open(csv_path) as f:
        reader = csv.DictReader(f, delimiter = ',')
        for row in reader:
            nrow += 1
            if nrow > maxrows: break
            texts.append(row['text'])
            stars.append(int(row['stars']))
            useful.append(int(row['useful']))
            funny.append(int(row['funny']))
            cool.append(int(row['cool']))
    return texts, stars, useful, funny, cool



## Stores a mapping of words to index positions
class WordVec:
    words_ix = dict()
    def __init__(self, word_list):
        self.word_vec = np.zeros(len(WordVec.words_ix))
        for word in word_list:
            self.word_vec[WordVec.words_ix[word]] += 1

    @classmethod
    def set_word_universe(cls, word_list):
        cls.words_ix = dict(zip(word_list, range(len(word_list))))

DATA_DIR = "../../yelp_analyses/data"
PUNCT_REMOVES = str.maketrans('', '', string.punctuation)
WS_COLLAPSE_RE = re.compile("\W+")

texts, stars, useful, funny, cool = \
    read_texts_stars(f'{DATA_DIR}/yelp_review.csv', 500)
word_lists = [get_words(text).split() for text in texts]
unique_words = list(set().union(*word_lists))

WordVec.set_word_universe(unique_words)
word_vecs = [WordVec(word_list) for word_list in word_lists]
## word_mat is a matrix with one column per data record and one
## row per feature. Features are counts of how many times each word
## appears.
word_mat = np.array([wv.word_vec for wv in word_vecs])
high_score = np.array([1 if rating >= 4 else 0 for rating in stars])


X_trn, y_trn, X_val, y_val, X_tst, y_tst = trn_val_tst(word_mat, high_score, 
                                                       8/10, 1/10, 1/10)

net_shape = [word_mat.shape[1], 20, 7, 5, 1]
activations = standard_binary_classification_layers(len(net_shape))

net = nn.Net(net_shape, activations, use_adam = True)
costs = net.train(X = X_trn.T, y = y_trn, 
                  iterations = 200, learning_rate = 0.1,
                  beta1 = 0.9, beta2 = 0.99,
                  minibatch_size = 64,
                  alpha_decayer = lambda t: 1 / (1 + t),
                  lambd = 0.6,
                  debug = True)
yhat_trn = net.predict(X_trn.T)
yyhat_trn = np.vstack((y_trn, yhat_trn)).T
auc_trn = roc_auc_score(y_trn, yhat_trn)

yhat_val = net.predict(X_val.T)
yyhat_val = bind_and_sort(y_val, yhat_val)
auc_val = roc_auc_score(y_val, yhat_val)
print("auc =", auc_val)





net_shape = [word_mat.shape[1], 30, 20, 20, 20, 20, 20, 1]
stars_vec = np.array([int(star) for star in stars])
activations = [avs.relu for i in net_shape]
X_trn, y_trn, X_val, y_val, X_tst, y_tst = trn_val_tst(word_mat, stars_vec, 
                                                       4/10, 3/10, 3/10)
stars_net = nn.Net(net_shape, 
                   activations = activations, 
                   loss = losses.MSE,
                   use_adam = True)
stars_net.train(X = X_trn.T, y = y_trn, 
                iterations = 50, 
                debug = True,
                lambd = 0.1,
                minibatch_size = 256,
                learning_rate = 0.005)
yhat_trn = stars_net.predict(X_trn.T)
yhat_val = stars_net.predict(X_val.T)

yyhat_trn = bind_and_sort(y_trn, yhat_trn)
yyhat_val = bind_and_sort(y_val, yhat_val)

plt.scatter(y_trn, yhat_trn)
plt.scatter(y_val, yhat_val)



