import unittest
from .align import align_sequences


class TestSequenceAlignment(unittest.TestCase):

    def setUp(self):
        self.sequence_a = 'the dog walks down the street'.split()
        self.sequence_b = 'the man walks up north through the street'.split()

    def test_align_string_sequences1(self):
        align1, align2, _ = align_sequences(self.sequence_a, self.sequence_b)
        self.assertEqual(' '.join(align1), 'the dog _ walks down _ _ _ the street')

    def test_align_string_sequences2(self):
        align1, align2, _ = align_sequences(self.sequence_a, self.sequence_b)
        self.assertEqual(' '.join(align2), 'the _ man walks _ up north through the street')

    def test_align_integer_sequences(self):
        sequence_a = [1, 1, 2, 1]
        sequence_b = [0, 1, 2, 0]
        align1, align2, _ = align_sequences(sequence_a, sequence_b)
        self.assertEqual(align1, [1, '_', 1, 2, 1, '_'])
        self.assertEqual(align2, ['_', 0, 1, 2, '_', 0])

    def test_align_iterator_sequences(self):
        sequence_a = [[1], [1], [2, 1]]
        sequence_b = [[0], [1], [2, 0]]
        align1, align2, _ = align_sequences(sequence_a, sequence_b)
        self.assertEqual(align1, [[1], ('_',), [1], [2, 1], ('_',)])
        self.assertEqual(align2, [('_',), [0], [1], ('_',), [2, 0]])

if __name__ == '__main__':
    unittest.main()
