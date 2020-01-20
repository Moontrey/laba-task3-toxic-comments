import glob
import io
from tqdm import tqdm
import pandas as pd

class DataLoading:

    def __init__(self):
        pass

    def many_folders(self, data_path, source_name, tags_file_name):
        tokens_dict = {'raw': [],
                   'pos': [],
                   'lemma': [],
                   'ner_target': [],
                   'word_net_pos': [],
                   'animacy_tag': [],
                   'title': [],
                   'date': [],
                   'genre': []
                   }
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

        tags_data = pd.DataFrame.from_dict(tokens_dict)
        return tags_data
