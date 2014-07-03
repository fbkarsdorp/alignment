from itertools import combinations

import pandas as pd


class Alignment(object):
    def __init__(self, sequences):
        assert all(len(sequence) == len(sequences[0]) for sequence in sequences[1:])
        self.alignment = pd.DataFrame(sequences)
        self.n_samples, self.n_features = self.alignment.shape

    def profile(self, plot=False):
        profile = self.alignment.apply(lambda x: x.value_counts() / x.count()).fillna(0.0)
        if plot:
            profile.T.plot(kind='bar')
        return profile

    def score(self, scoring_fn=None, gap_weight=1.0):
        """
        Return the average sum of pairs score over all columns.
        """
        if scoring_fn is None:
            scoring_fn = lambda a, b: 0.0 if a == b else 2.0
        scores = []
        for column in self.alignment.columns:
            score = 0.0
            count = 0.0
            for val_a, val_b in combinations(self.alignment[column].values, 2):
                if val_a == '_' or val_b == '_':
                    count += gap_weight
                else:
                    score += scoring_fn(val_a, val_b)
                    count += 1.0
            scores.append(score / count)
        return sum(scores) / len(scores)

    def plot(self):
        """
        Plot the alignment using matplotlib and seaborn.
        """
        import seaborn as sb
        # assign to each unique value in the alignment as unique color
        unique_values = list(set(value for row in self.alignment.values for value in row))
        colors = dict(zip(unique_values, sb.color_palette("husl", len(unique_values))))
        # plot all the alignments
        fig, ax = sb.plt.subplots(figsize=(self.n_features * 1.5, self.n_samples))
        for i, row in self.alignment.iterrows():
            for j, value in enumerate(row.values):
                ax.annotate(value, xy=(j, self.n_samples - i - 1), color=colors[value])
        # remove some default plotting behaviour of matplotlib
        ax.set_yticks(range(self.n_samples))
        ax.set_yticklabels(self.alignment.index)
        ax.set_xticklabels(())
        ax.set_ylim(-1, self.n_samples)
        ax.set_xlim(-1, self.n_features)

    def __repr__(self):
        return '<Alignment of %d sequences>' % self.n_samples

    def __str__(self):
        return self.alignment.to_string()