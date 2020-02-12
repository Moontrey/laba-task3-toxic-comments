import re
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt

class Visualisation:

    def __init__(self, data_frame):

        self.data = data_frame

    def _bar_plot(self, ax, x, y, title):
        sns.barplot(data=self.data, x=x,
                    y=y, ax=ax)
        ax.set_title(title)
        for item in ax.get_xticklabels():
            item.set_rotation(45)

    def pre_process_tags(self, skills_df):
        concatenated_tags = ' '.join(list(skills_df.columns))
        concatenated_tags = re.sub('[^&.#+\w \t_]', '_', concatenated_tags)
        concatenated_tags = re.sub('microsoft', 'ms', concatenated_tags)
        concatenated_tags = re.sub('js', '_javascript_', concatenated_tags)
        concatenated_tags = re.sub(' git', ' git_', concatenated_tags)
        concatenated_tags = re.sub('git ', '_git ', concatenated_tags)
        concatenated_tags = re.sub('c\+\+', 'c\+_', concatenated_tags)
        concatenated_tags = re.sub('_+', '_', concatenated_tags)
        concatenated_tags = re.sub('_ | _', ' ', concatenated_tags)
        return concatenated_tags.split(' '), concatenated_tags

    def popular_skills(self):
        sum_of_skills = pd.DataFrame(self.data.iloc[:, 3:].sum(),
                                     columns=['Number']
                                     ).sort_values(by='Number', ascending=False)[:30]

        f = plt.figure(figsize=(20, 8))
        self._bar_plot(sum_of_skills, x=list(sum_of_skills.index), y=sum_of_skills.Number, ax=f.gca(),
                       title='Top 30 popular skills')
