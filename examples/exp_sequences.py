from alignment import align_sequences
from alignment import Alignment

from alignment.utils import merge

sequence_a = 'voldemort'
sequence_b = 'waldemort'

# align the two sequences
align_a, align_b, distance = align_sequences(sequence_a, sequence_b)
# construct a new Alignment object
alignment = Alignment.from_sequences(align_a, align_b)
# pretty print the alignment
print alignment

