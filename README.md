# (Multiple) Sequence Alignment

## Examples

``` python
from alignment import multi_sequence_alignment

sequences = ['the quick fox jumps over the dog'.split(),
             'the brown fox jumps over the lazy dog'.split(),
             'the clever fox jumps over the lazy crow'.split()]
alignment = multi_sequence_alignment(sequences)
print(alignment)
print(alignment.score())
     0       1    2      3     4    5     6     7
0  the  clever  fox  jumps  over  the  lazy  crow
1  the   quick  fox  jumps  over  the     _   dog
2  the   brown  fox  jumps  over  the  lazy   dog
0.29166666666666663
```
