# (Multiple) Sequence Alignment

## Examples

``` python
from alignment import multi_sequence_alignment

sequences = ['the quick fox jumps over the dog'.split(),
             'the brown fox jumps over the lazy dog'.split(),
             'the clever fox jumps over the lazy crow'.split()]
alignment = multi_sequence_alignment(sequences, gap_weight=0, gap_penalty=6)
print(alignment)
print(alignment.score())
     0       1    2      3     4    5     6     7
0  the   quick  fox  jumps  over  the     _   dog
1  the   brown  fox  jumps  over  the  lazy   dog
2  the  clever  fox  jumps  over  the  lazy  crow
0.29166666666666663
```
