# SYDE532 Kuramoto Project

Kuramoto oscillator simulation for 2D cortical sheets.

## Installation
pip install -r requirements.txt
# or
pip install -e .

## Notes:
N = # nodes
Nodal data as vectors: [N,]
Edge data as matrices: [N,N]

WIP:
1) Unify coupling definition apis 
    - strength, kernel weight, mask for all cases specified by component list
    - Convert to jax only
    - Get rid of coupling mode -> only use groups / masks
