# SYDE532 Kuramoto Project

Kuramoto oscillator simulation for 2D cortical sheets.

## Installation
pip install -r requirements.txt

or

pip install -e .

## Notes:
N = # nodes

Nodal data as vectors: [N,]

Edge data as matrices: [N,N]

WIP:
1) Coupling specification

### Coupling components
`coupling.components` is a list of components. Each component is configured by:

- `kernel`: kernel name (currently `gaussian` is differentiable in JAX; `constant` is supported for uniform coupling)
- `base_strength`: scalar multiplier for the component
- `radius`: optional distance cutoff
- `kernel_params`: dict of kernel parameters
- `node_groups`: optional list of group ids for node selection
- `edge_mode`: one of `within`, `outgoing`, `incoming`, `custom`
  - `within`: sender and receiver are both in `node_groups`
  - `outgoing`: sender is in `node_groups`, receiver is any node (default)
  - `incoming`: receiver is in `node_groups`, sender is any node
  - `custom`: sender is in `node_groups` and receiver is in `to_node_groups`
- `to_node_groups`: optional receiver group list, used only with `edge_mode="custom"`
