from sklearn_crfsuite import CRF, scorers, metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm_notebook
from hmmlearn import hmm


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

    def fit(self, dataframe, gamma=0):

        self.dataframe = dataframe
        self.tags = dataframe['tags'].unique().tolist()
        self.list_of_tags = dataframe['tags'].values.tolist()
        self.list_of_words = list(np.append(dataframe['raw'].unique(), np.array(["<UNK>"])))

        self.transmat_ = self.transition_(gamma=gamma)
        self.emissionprob_ = self.emission_()
        self.startprob_ = self.initial_state()
        self.n_components = len(self.tags)

        return self

    def transition_(self, gamma=0):
        '''
        Transition probability matrix: probability of tag after tag
        :return: transition probability matrix
        '''
        transition = np.zeros((len(self.tags), len(self.tags)))
        for previous, current in zip(self.list_of_tags, self.list_of_tags[1:]):
            transition[self.tags.index(previous)][self.tags.index(current)] += 1
        if gamma:
            transition = self.smoothing(transition, gamma=gamma)

        return transition

    def smoothing(self, transition, gamma=0):

        n_words = len(np.unique(self.list_of_words))
        for i in range(len(transition)):
            transition[i] = (transition[i] + gamma) / (sum(transition[i]) + gamma * n_words)
            transition[i] = transition[i] / sum(transition[i])

        return transition

    def emission_(self):
        '''
        Emission probability matrix: probability of word given tag
        :return: emission matrix
        '''

        emission = np.zeros((len(self.tags), len(self.list_of_words)))
        for i in tqdm_notebook(zip(self.dataframe['raw'], self.dataframe['tags'])):
            emission[self.tags.index(i[1])][self.list_of_words.index(i[0])] += 1

        for i in range(len(emission)):
            emission[i] = emission[i] / sum(emission[i])

        return emission

    def initial_state(self):
        '''
        Making an array with distribution of a first tag
        :return: an array with probabilities of a first tag
        '''
        tag_first_in_sent = dict(self.dataframe.groupby(['n_sent']).first().groupby(['tags'])['raw'].count())
        full_tags_list = list({k: (tag_first_in_sent[k] if k in tag_first_in_sent.keys() else 0) for k in self.tags}.values())
        prob = full_tags_list / sum(full_tags_list)

        return prob



