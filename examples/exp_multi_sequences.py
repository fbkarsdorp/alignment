from alignment import multi_sequence_alignment


sequence_a = 'the quick fox jumps over the dog'.split()
sequence_b = 'the brown fox jumps over the lazy dog'.split()
sequence_c = 'the clever fox jumps over the lazy crow'.split()

# align the three sequences using progressive multiple alignment
alignment = multi_sequence_alignment([sequence_a, sequence_b, sequence_c])

# pretty print the alignment
print alignment

# compute the alignment score
print alignment.score()

