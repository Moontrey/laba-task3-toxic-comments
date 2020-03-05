from sklearn_crfsuite import CRF, scorers, metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm_notebook
from hmmlearn import hmm
import json
import pickle
from future.utils import iteritems

def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {:d} samples in lengths array {!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]


def custom_t_t_split(X, lengths, test_size=0.4, random_state=77):
    corpus = []
    for i, j in iter_from_X_lengths(X, lengths):
        corpus.append(X[i:j].tolist())
    X_train, X_test, l_train, l_test = train_test_split(corpus, lengths, test_size=test_size,
                                                        shuffle=False, random_state=random_state)
    X_train = [item for sublist in X_train for item in sublist]
    X_test = [list(item) for sublist in X_test for item in sublist]
    return np.array(X_train), np.array(X_test), l_train, l_test


class CrfFeatures:

    def __init__(self):
        pass

    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1]
        # lemma = sent[i][3]
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
            'has_hyphen': '-' in word,
            # 'has_dot': '.' in word,
            # 'lemma': lemma,
        }
        if i > 0:
            word1 = sent[i - 1][0]
            postag1 = sent[i - 1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            postag1 = sent[i + 1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    def sent2labels(self, sent):
        return [label for token, postag, label, lemma in sent]

    def run(self, sents):
        X = [self.sent2features(s) for s in sents]
        y = [self.sent2labels(s) for s in sents]
        return X, y


class HMM(hmm.MultinomialHMM):

    def __init__(self, n_components, algorithm="viterbi", gamma=0):
        super().__init__(self, n_components,
                         algorithm=algorithm)
        self.gamma = gamma
        self.n_components = n_components

    def fit(self, w_t, tags, lengths):

        self.vocab = list(np.append(np.unique(w_t[:, 0]), np.array(["<UNK>"])))

        self.transmat_ = self.transition_(w_t, tags, lengths)
        self.emissionprob_ = self.emission_(w_t, tags)
        self.startprob_ = self.initial_state(w_t, tags, lengths)

        return self

    def transition_(self, w_t, tags, lengths):
        '''
        Transition probability matrix: probability of tag after tag
        :return: transition probability matrix
        '''
        transition = np.zeros((len(tags), len(tags)))

        for i, j in iter_from_X_lengths(w_t, lengths):
            for previous, current in zip(w_t[i:j][:, 1], w_t[i + 1:j][:, 1]):
                transition[tags.index(previous)][tags.index(current)] += 1
        if self.gamma:
            transition = self.smoothing(transition)
        else:
            transition = np.array([i/np.sum(i) if np.sum(i) != 0 else np.zeros(len(i)) for i in transition])
        return transition

    def smoothing(self, to_smooth):

        n = to_smooth.shape[1]
        for i in range(len(to_smooth)):
            to_smooth[i] = (to_smooth[i] + self.gamma) / (np.sum(to_smooth[i]) + self.gamma * n)
            to_smooth[i] = to_smooth[i] / np.sum(to_smooth[i])
        return to_smooth

    def emission_(self, w_t, tags):
        '''
        Emission probability matrix: probability of word given tag
        :return: emission matrix
        '''
        self.vocab = list(np.append(np.unique(w_t[:, 0]), np.array(["<UNK>"])))
        emission = np.zeros((len(tags), len(self.vocab)))
        for i in tqdm_notebook(w_t):
            emission[tags.index(i[1])][self.vocab.index(i[0])] += 1
        if self.gamma:
            emission = self.smoothing(emission)
        else:
            for i in range(len(emission)):
                emission[i] = emission[i] / sum(emission[i])
        return emission

    def initial_state(self, w_t, tags, lengths):
        '''
        Making an array with distribution of a first tag
        :return: an array with probabilities of a first tag
        '''
        tag_first_in_sent = np.zeros(len(tags))
        for i, _ in iter_from_X_lengths(w_t, lengths):
            tag_first_in_sent[tags.index(w_t[i][1])] += 1
        prob = tag_first_in_sent / np.sum(tag_first_in_sent)

        return prob

    def predict(self, X, tags, lengths=None):
        # prepare data for HMM
        X_new = X[:, 0]
        for i in tqdm_notebook(range(len(X_new))):
            if X_new[i] not in self.vocab:
                X_new[i] = "<UNK>"
        word2idx = self.word2idx_()
        X_new = np.array([word2idx[word] for word in X_new]).reshape(-1, 1)
        print('Decoding started')
        _, state_sequence = super().decode(X_new, lengths)
        idx2tag = self.idx2tag_(tags)
        decoded_predicts = [idx2tag[key] for key in state_sequence]
        return decoded_predicts

    def word2idx_(self):
        return {w: i for i, w in enumerate(self.vocab)}

    def tag2idx_(self, tags):
        return {t: i for i, t in enumerate(tags)}

    def idx2tag_(self, tags):
        return {v: k for k, v in iteritems(self.tag2idx_(tags))}


class SpacyFit:
    # Convert .tsv file to dataturks json format.

    def __init__(self):
        pass

    def tsv_to_json(self, input_path, output_path, unknown_label):

        f = open(input_path, 'r', encoding='utf-8')  # input file
        fp = open(output_path, 'w')  # output file
        data_dict = {}
        annotations = []
        label_dict = {}
        s = ''
        start = 0
        for line in f:
            if line[0:len(line) - 1] != '.\tO':
                word, entity = line.split('\t')
                s += word + " "
                entity = entity[:len(entity) - 1]
                if entity != unknown_label:
                    if len(entity) != 1:
                        d = dict()
                        d['text'] = word
                        d['start'] = start
                        d['end'] = start + len(word) - 1
                        try:
                            label_dict[entity].append(d)
                        except:
                            label_dict[entity] = []
                            label_dict[entity].append(d)
                start += len(word) + 1
            else:
                data_dict['content'] = s
                s = ''
                label_list = []
                for ents in list(label_dict.keys()):
                    for i in range(len(label_dict[ents])):
                        if label_dict[ents][i]['text'] != '':
                            l = [ents, label_dict[ents][i]]
                            for j in range(i + 1, len(label_dict[ents])):
                                if label_dict[ents][i]['text'] == label_dict[ents][j]['text']:
                                    di = dict()
                                    di['start'] = label_dict[ents][j]['start']
                                    di['end'] = label_dict[ents][j]['end']
                                    di['text'] = label_dict[ents][i]['text']
                                    l.append(di)
                                    label_dict[ents][j]['text'] = ''
                            label_list.append(l)

                for entities in label_list:
                    label = dict()
                    label['label'] = [entities[0]]
                    label['points'] = entities[1:]
                    annotations.append(label)
                data_dict['annotation'] = annotations
                annotations = []
                json.dump(data_dict, fp)
                fp.write('\n')
                data_dict = {}
                start = 0
                label_dict = {}

    def spacy_corpus(self, input_file=None, output_file=None):

        training_data = []
        with open(input_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                point = annotation['points'][0]
                labels = annotation['label']
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    entities.append((point['start'], point['end'] + 1, label))

            training_data.append((text, {"entities": entities}))

        with open(output_file, 'wb') as fp:
            pickle.dump(training_data, fp)

        return training_data

    def run(self, data):

        data.to_csv('ner_corpus.tsv', sep='\t', index=False, header=False)
        self.tsv_to_json("ner_corpus.tsv", 'ner_corpus.json', 'abc')
        spacy_input = self.spacy_corpus(input_file='ner_corpus.json', output_file='ner_corpus')
        spacy_input = [sent for sent in spacy_input if sent[0] != '']
        return spacy_input



