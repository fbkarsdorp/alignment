from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import merge


plt.style.use('seaborn')


class Alignment(object):
    def __init__(self, sequences):
        assert all(len(sequence) == len(sequences[0]) for sequence in sequences[1:])
        self.alignment = pd.DataFrame(sequences)
        self.n_samples, self.n_features = self.alignment.shape

    @classmethod
    def from_sequences(cls, *sequences):
        return Alignment(merge(sequences))

    def profile(self, plot=False):
        profile = self.alignment.apply(lambda x: x.value_counts() / x.count()).fillna(0.0)
        if plot:
            profile.T.plot(kind='bar')
        return profile

    def score(self, scoring_fn=None, gap_weight=1.0, gap_penalty=1.0, per_column=False):
        """
        Return the average sum of pairs score over all columns.
        """
        if scoring_fn is None:
            scoring_fn = lambda a, b: 0.0 if a == b else 2.0
        scores = []
        for column in self.alignment.columns:
            score = 0.0
            count = 0.0
            for i in range(self.n_samples):
                for j in range(self.n_samples):
                    if i != j:
                        val_a, val_b = self.alignment[column].values[i], self.alignment[column].values[j]
                        if val_a == '_' and val_b == '_':
                            score += 0.0
                        elif val_a == '_' or val_b == '_':
                            score += gap_weight
                        else:
                            score += 0.0 if scoring_fn(val_a, val_b) < 1 else 1.0
                        count += 1.0
            scores.append(score / count)
        return scores if per_column else sum(scores) / len(scores)

    def _row_based_score(self, scoring_fn=None, gap_weight=1.0, gap_penalty=1.0):
        if scoring_fn is None:
            scoring_fn = lambda a, b: 0.0 if a == b else 2.0
        scores = []
        for i, row in self.alignment.iterrows():
            row_score = []
            for column in self.alignment.columns:
                if row[column] != '_':
                    score, count = 0, 0
                    for j in range(self.n_samples):
                        if j != i:
                            val_a, val_b = self.alignment.ix[i, column], self.alignment.ix[j, column]
                            if val_a == '_' or val_b == '_':
                                score += gap_weight
                            else:
                                score += 0.0 if scoring_fn(val_a, val_b) < 1 else 1.0
                            count += 1.0
                    row_score.append(score / count)
            scores.extend([(pos / len(row_score), score) for pos, score in enumerate(row_score, 1)])
        return scores

    def plot(self, filepath=None):
        """
        Plot the alignment using matplotlib.
        """
        # assign to each unique value in the alignment as unique color
        unique_values = list(set(value for row in self.alignment.values for value in row))
        colors = iter(plt.cm.get_cmap('Set3')(np.linspace(0, 1, len(unique_values))))
        colors = dict(zip(unique_values, colors))
        # plot all the alignments
        fig, ax = plt.subplots(figsize=(self.n_features * 1.5, self.n_samples))
        for i, row in self.alignment.iterrows():
            for j, value in enumerate(row.values):
                ax.annotate(value, xy=(j, self.n_samples - i - 1), color=colors[value])
        # remove some default plotting behaviour of matplotlib
        ax.set_yticks(range(self.n_samples))
        ax.set_yticklabels(self.alignment.index)
        ax.set_xticklabels(())
        ax.set_ylim(-1, self.n_samples)
        ax.set_xlim(-1, self.n_features)
        if filepath:
            plt.savefig(filepath + (".pdf" if not filepath.endswith('.pdf') else ''))
            plt.close(fig)

    def __repr__(self):
        return f'<Alignment of {self.n_samples} sequences>'

    def __str__(self):
        return self.alignment.to_string()
