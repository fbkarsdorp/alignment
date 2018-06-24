from functools import partial

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from .alignment import Alignment
from .utils import flatten, merge


NONE, LEFT, UP, DIAG = 0, 1, 2, 3


def needle_wunsch(sequence_a, sequence_b, scorer, gap_penalty=1, scale=1.0):
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
    length = np.zeros((len1 + 1, len2 + 1), dtype='f')
    pointer[0, 0] = NONE
    pointer[0, 1:] = LEFT
    pointer[1:, 0] = UP
    for i in range(1, len1 + 1):
        matrix[i, 0] = matrix[i - 1, 0] + gap_penalty * scale
        length[i, 0] = length[i - 1, 0] + gap_penalty * scale
    for j in range(1, len2 + 1):
        matrix[0, j] = matrix[0, j - 1] + gap_penalty * scale
        length[0, j] = length[0, j - 1] + gap_penalty * scale
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            gap_a = matrix[i - 1, j] + (gap_penalty * scale if pointer[i - 1, j] == UP else gap_penalty)
            gap_b = matrix[i, j - 1] + (gap_penalty * scale if pointer[i, j - 1] == LEFT else gap_penalty)
            match = matrix[i - 1, j - 1] + scorer[i - 1, j - 1]
            if gap_a < match and gap_a <= gap_b:
                matrix[i, j] = gap_a
                pointer[i, j] = UP
            elif match <= gap_b:
                matrix[i, j] = match
                pointer[i, j] = DIAG
            else:
                matrix[i, j] = gap_b
                pointer[i, j] = LEFT
            p = pointer[i, j]
            l_gap_a = length[i - 1, j] + (gap_penalty * scale if p == UP else 0)
            l_gap_b = length[i, j - 1] + (gap_penalty * scale if p == LEFT else 0)
            l_match = length[i - 1, j - 1] + (scorer[i - 1, j - 1] if p == DIAG else 0)
            length[i, j] = max(l_gap_a, l_gap_b, l_match)
    # normalize the distance
    distance = matrix[len1, len2] / length[len1, len2]
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


def align(sequence_a, sequence_b, scores, gap_penalty=1, scale=1.0):
    matrix, pointer, distance = needle_wunsch(
        sequence_a, sequence_b, scores, gap_penalty, scale)
    align1, align2 = backtrace(pointer, sequence_a, sequence_b)
    return align1, align2, distance


def align_sequences(sequence_a, sequence_b, scoring_fn=None, gap_penalty=1, scale=1.0):
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


def _align_profiles(sequence_a, sequence_b, scoring_fn=None,
                    gap_penalty=1, scale=1.0, gap_weight=0.5):
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
                        dist += gap_weight
                    else:
                        dist += 0.0 if scoring_fn(mi, mj) < 1 else 1.0
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


def multi_sequence_alignment(sequences, scoring_fn=None, linking='single',
                             gap_penalty=1, scale=1.0, gap_weight=1.0, verbosity=0):
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
    matrix = pairwise_distances(sequences, partial(align_sequences, scoring_fn=scoring_fn,
                                                   gap_penalty=gap_penalty, scale=scale))
    # compute the guiding tree to do the progressive alignment
    Z = linkage(squareform(matrix), method='single')
    # perform the alignment by iterating through the clusters
    alignments = {}
    n_seqs = len(sequences)
    for cluster_id, (node1, node2, _, _) in enumerate(Z, n_seqs):
        node1, node2 = int(node1), int(node2)
        if node1 < n_seqs and node2 < n_seqs:
            align1, align2, _ = align_sequences(sequences[node1], sequences[node2],
                                                scoring_fn, gap_penalty, scale)
        else:
            if node1 < n_seqs:
                sequence_a, sequence_b = [[elt] for elt in sequences[node1]], alignments[node2]
            elif node2 < n_seqs:
                sequence_a, sequence_b = alignments[node1], [[elt] for elt in sequences[node2]]
            else:
                sequence_a, sequence_b = alignments[node1], alignments[node2]
            align1, align2, _ = _align_profiles(sequence_a, sequence_b, scoring_fn, gap_penalty, scale, gap_weight)
        alignments[cluster_id] = merge(align1, align2)
    return Alignment(list(zip(*map(flatten, alignments[max(alignments)]))))

if __name__ == '__main__':
    sequences = ['the quick fox jumps over the dog'.split(),
                 'the brown fox jumps over the lazy dog'.split(),
                 'the clever fox jumps over the lazy crow'.split()]
    alignment = multi_sequence_alignment(sequences)
    print(alignment)
    print(alignment.score())
    alignment.plot("testje.pdf")
