from sklearn_crfsuite import CRF, scorers, metrics
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm_notebook
from hmmlearn import hmm
import json
import pickle


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



