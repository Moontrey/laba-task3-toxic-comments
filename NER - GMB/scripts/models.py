from sklearn_crfsuite import CRF, scorers, metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm_notebook

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


class HMM:

    def __init__(self, dataframe):
        '''
        :param dataframe: dataframe with column 'raw' (corpus tokens)
        '''
        self.dataframe = dataframe
        self.tags = dataframe['tags'].unique().tolist()
        self.list_of_tags = dataframe['tags'].values.tolist()
        self.list_of_words = dataframe['raw'].unique().tolist()

    def transition_(self):
        '''
        Transition probability matrix: probability of tag after tag
        :return: transition probability matrix
        '''
        transition = np.zeros((len(self.tags), len(self.tags)))
        for i in range(len(self.tags)):
            for j in range(0, len(self.list_of_tags) - 1):
                if self.list_of_tags[j] == self.tags[i]:
                    if self.list_of_tags[j] == self.list_of_tags[j + 1]:
                        transition[i][i] += 1
                    else:
                        transition[i][self.tags.index(self.list_of_tags[j + 1])] += 1
        for i in range(len(transition)):
            transition[i] = transition[i] / np.count_nonzero(transition[i])

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
            emission[i] = emission[i] / np.count_nonzero(emission[i])

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

    def run(self):

        return self.transition_(), self.emission_(), self.initial_state()
