import glob
import io
from tqdm import tqdm
import pandas as pd

class DataLoading:


    def __init__(self):
        pass

    def many_folders(self, data_path, source_name, tags_file_name, columns):
        '''
        :param data_path: path to folder with folders
        :param source_name: related file name
        :param tags_file_name: related file name
        :param columns: related columns in returned dataframe

        :return: dataframe with data from all files
        '''
        tokens_dict = {key: [] for key in columns}
        n_sent = 0
        for folder in tqdm(glob.iglob(rf"{data_path}\*\*", recursive=False)):
            with io.open(folder + source_name, 'r', encoding='utf-8') as source:
                try:
                    met_lines = [line.split(': ')[1].strip() for line in source.readlines()]
                except Exception as e:
                    print(e, source.readlines())

                if met_lines[4] != 'Voice of America':
                    continue

                with io.open(folder + tags_file_name, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        splitted = line.split('\t')
                        if len(splitted) <= 1:
                            n_sent += 1
                            continue

                        tokens_dict['title'].append(met_lines[0])
                        tokens_dict['date'].append(met_lines[1])
                        tokens_dict['genre'].append(met_lines[3])

                        tokens_dict['raw'].append(splitted[0])
                        tokens_dict['pos'].append(splitted[1])
                        tokens_dict['lemma'].append(splitted[2])
                        tokens_dict['ner_target'].append(splitted[3])
                        tokens_dict['word_net_pos'].append(splitted[4])
                        tokens_dict['animacy_tag'].append(splitted[7])
                        tokens_dict['n_sent'].append(n_sent)

        tags_data = pd.DataFrame.from_dict(tokens_dict)
        tags_data['ner_target'] = tags_data['ner_target'].apply(lambda x: x.split('-')[0])
        return tags_data

    def bio_tags(self, annotated_sentence):
        """
        Transform a pseudo-IOB notation to proper IOB notation

        `annotated_sentence` = list of triplets [(w1, t1, iob1), ...]

        return: annotated_sentence with proper IOB notation
        """
        proper_iob_tokens = []
        for idx, annotated_token in enumerate(annotated_sentence):
            tag, word, ner = annotated_token
            if ner != 'O':
                if idx == 0:
                    ner = "B-" + ner
                elif annotated_sentence[idx - 1][2] == ner:
                    ner = "I-" + ner
                else:
                    ner = "B-" + ner
            proper_iob_tokens.append((tag, word, ner))
        return proper_iob_tokens

    def get_sent(self, data):
        '''creating lists of triplets [(w1, t1, iob1), ...]
        data: data frame

        return: lists of triplets [(w1, t1, iob1), ...]
        '''

        data = data

        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["raw"].values.tolist(),
                                                           s["pos"].values.tolist(),
                                                           s["ner_target"].values.tolist())]
        grouped = data.groupby("n_sent").apply(agg_func)
        sentences = [s for s in grouped]
        sentences = [self.bio_tags(sent) for sent in sentences]
        return sentences
