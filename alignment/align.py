from functools import partial
from itertools import combinations

import numpy as np
import pandas as pd

from HACluster import Clusterer, single_link
from utils import flatten


NONE, LEFT, UP, DIAG = 0, 1, 2, 3


def compute_matrix(sequence_a, sequence_b, scorer, gap_penalty=1, scale=0.5):
    """
    :param sequence_a: any iterable with a fixed order.
    :param sequence_b: any iterable with a fixed order.
    :param scorer: a dictionary holding the scores between all pairwise
                   items in sequence_a and sequence_b.
    :param gap_penalty: the gap opening penalty used in the analysis.
    :param scale: the factor by which gap_penalty should be decreased.
    :return: numpy matrix, backtrace pointers and the distance between the
             two sequences.
    """
    len1, len2 = len(sequence_a), len(sequence_b)
    pointer = np.zeros((len1 + 1, len2 + 1), dtype='i')
    matrix = np.zeros((len1 + 1, len2 + 1), dtype='f')
    pointer[0, 0] = NONE
    pointer[0, 1:] = LEFT
    pointer[1:, 0] = UP
    for i in range(1, len1 + 1):
        matrix[i, 0] = matrix[i - 1, 0] + gap_penalty * scale
    for j in range(1, len2 + 1):
        matrix[0, j] = matrix[0, j - 1] + gap_penalty * scale
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if pointer[i - 1, j] == UP:
                gap_a = matrix[i - 1, j] + gap_penalty * scale
            else:
                gap_a = matrix[i - 1, j] + gap_penalty
            if pointer[i, j - 1] == LEFT:
                gap_b = matrix[i, j - 1] + gap_penalty * scale
            else:
                gap_b = matrix[i, j - 1] + gap_penalty
            match = matrix[i - 1, j - 1] + scorer[i - 1, j - 1]
            if gap_a < match and gap_a < gap_b:
                matrix[i, j] = gap_a
                pointer[i, j] = UP
            elif match < gap_b:
                matrix[i, j] = match
                pointer[i, j] = DIAG
            else:
                matrix[i, j] = gap_b
                pointer[i, j] = LEFT
    distance = matrix[len1, len2] / (max(len1, len2) + gap_penalty * scale)
    return matrix, pointer, distance


def backtrace(pointer, sequence_a, sequence_b):
    i, j = len(sequence_a), len(sequence_b)
    align1, align2 = [], []
    fill_a, fill_b = '_', '_'
    if any(isinstance(e, (tuple, list)) for e in sequence_a):
        fill_a = ('_',) * len(sequence_a[0])
    if any(isinstance(e, (tuple, list)) for e in sequence_b):
        fill_b = ('_',) * len(sequence_b[0])
    while True:
        p = pointer[i, j]
        if p == NONE:
            break
        if p == DIAG:
            align1.append(sequence_a[i - 1])
            align2.append(sequence_b[j - 1])
            i, j = i - 1, j - 1
        elif p == LEFT:
            align1.append(fill_a)
            align2.append(sequence_b[j - 1])
            j -= 1
        elif p == UP:
            align1.append(sequence_a[i - 1])
            align2.append(fill_b)
            i -= 1
        else:
            raise ValueError("Something went terribly wrong.")
    return align1[::-1], align2[::-1]


def align(sequence_a, sequence_b, scores, gap_penalty=1, scale=0.5):
    matrix, pointer, distance = compute_matrix(
        sequence_a, sequence_b, scores, gap_penalty, scale)
    align1, align2 = backtrace(pointer, sequence_a, sequence_b)
    return align1, align2, distance


def align_sequences(sequence_a, sequence_b, scoring_fn=None, gap_penalty=1, scale=0.5):
    """
    Align two sequences using the Needleman-Wunsch algorithm.

    :param sequence_a: some fixed order iterable.
    :param sequence_b: some fixed order iterable.
    :param scoring_fn: a distance function.
    :param gap_penalty: the penalty for inserting gaps.
    :param scale: the factor by which gap_penalty should be decreased.
    :return: two new sequences with gaps inserted and the distance between them.
    """
    if scoring_fn is None:
        scoring_fn = lambda a, b: 0.0 if a == b else 2.0
    scores = {(i, j): scoring_fn(sequence_a[i], sequence_b[j])
              for i in range(len(sequence_a)) for j in range(len(sequence_b))}
    return align(sequence_a, sequence_b, scores, gap_penalty, scale)


def _align_profiles(sequence_a, sequence_b, scoring_fn=None, gap_penalty=1, scale=0.5, gap_weight=1.0):
    scores = {}
    for i in range(len(sequence_a)):
        for j in range(len(sequence_b)):
            dist = 0.0
            count = 0.0
            for k in range(len(sequence_a[i])):
                for l in range(len(sequence_b[j])):
                    mi = sequence_a[i][k]
                    mj = sequence_b[j][l]
                    if mi == '_' or mj == '_':
                        count += gap_weight
                    else:
                        dist += scoring_fn(mi, mj)
                        count += 1.0
                scores[i, j] = dist / count
    return align(sequence_a, sequence_b, scores, gap_penalty, scale)


def pairwise_distances(sequences, fn):
    distances = np.zeros((len(sequences), len(sequences)))
    for i in range(len(sequences)):
        for j in range(i):
            _, _, distance = fn(sequences[i], sequences[j])
            distances[i, j] = distance
            distances[j, i] = distances[i, j]
    return distances


def multi_sequence_alignment(sequences, scoring_fn=None, linkage=single_link):
    """
    Perform progressive multiple sequence alignment.

    :param sequences: some iterable of fixed order iterables.
    :param scoring_fn: a distance function.
    :param linkage: the linkage function to use for the clustering.
    :return: an Alignment object.
    """
    if scoring_fn is None:
        scoring_fn = lambda a, b: 0.0 if a == b else 2.0
    # compute all pairwise distances
    matrix = pairwise_distances(sequences, partial(align_sequences, scoring_fn=scoring_fn))
    # compute the guiding tree to do the progressive alignment
    clusterer = Clusterer(matrix, linkage=linkage)
    clusterer.cluster()
    # perform the alignment by iterating through the clusters
    alignments = {}
    n_seqs = len(sequences)
    for cluster_id, node1, node2, _, _ in clusterer.dendrogram():
        if node1 < n_seqs and node2 < n_seqs:
            align1, align2, _ = align_sequences(sequences[node1], sequences[node2], scoring_fn)
        else:
            if node1 < n_seqs:
                sequence_a, sequence_b = [[elt] for elt in sequences[node1]], alignments[node2]
            elif node2 < n_seqs:
                sequence_a, sequence_b = alignments[node1], [[elt] for elt in sequences[node2]]
            else:
                sequence_a, sequence_b = alignments[node1], alignments[node2]
            align1, align2, _ = _align_profiles(sequence_a, sequence_b, scoring_fn)
        alignments[cluster_id] = map(flatten, zip(align1, align2))
    return Alignment(zip(*map(flatten, alignments[max(alignments)])))


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


if __name__ == '__main__':
    sequences = ['the quick fox jumps over the dog'.split(),
                 'the brown fox jumps over the lazy dog'.split(),
                 'the clever fox jumps over the lazy crow'.split()]
    alignment = multi_sequence_alignment(sequences)
    print alignment
    print alignment.score()