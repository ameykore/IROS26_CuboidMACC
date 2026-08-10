# Cuboid-MACC

Implementation for the paper:

**Hierarchical Planning for the Cuboid Multi-Agent Collective Construction (Cuboid-MACC) Problem**

Accepted at the **2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026)**.

---

## Overview

The Multi-Agent Collective Construction (MACC) problem studies how a team of mobile robots can cooperatively assemble large structures from simple building blocks. This repository presents **Cuboid-MACC**, which extends the original MACC formulation from unit cubic blocks to heterogeneous cuboid blocks. Supporting cuboid blocks allows the planner to represent richer structures, including hollow regions, bridges, cantilevers, roofs, and other overhanging features, while introducing new planning challenges arising from block orientation and carried-block collisions.

To address these challenges, Cuboid-MACC uses a hierarchical planning framework consisting of:

1. High-level abstract action planning using A*
2. Reachability analysis for abstract actions
3. Action Dependency Graph (ADG) generation
4. Multi-robot task allocation
5. Low-level multi-agent path planning using a modified Conflict-Based Search (CBS)

This decomposition reduces the planning burden by separating abstract construction ordering from low-level multi-robot path planning, while checking reachability and collision constraints under the Cuboid-MACC model.

---

## Repository Structure

```
IROS26_CuboidMACC
│
├── src/                  Main implementation
├── scripts/              Scripts for experiment analysis and statistics
├── results/              Example experiment outputs
├── legacy/               Previous development versions
├── papers/               Manuscript-related files, if included
├── environment.yml       Conda environment
└── README.md
```

The primary implementation is located in `src`.

The source code is organized into the following modules:

| Folder | Description |
|---------|-------------|
| assembly | High-level planning and reachable abstract actions |
| assembly/cpp | C++ acceleration modules for reachability computation |
| parallelization | Action Dependency Graph generation, robot assignment, and interface between planning levels |
| path | Low-level multi-agent path planning using modified CBS |
| utility | Shared utility functions |
| visualize | MeshCat visualization and animation utilities |
| data | Benchmark structures used in the experiments |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/junilm24/IROS26_CuboidMACC.git
cd IROS26_CuboidMACC
```

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate <environment_name>
```

The environment includes the required Python packages, including:

- numpy
- scipy
- numba
- networkx
- pyyaml
- dill
- matplotlib
- meshcat
- trimesh
- plotly

---

## Compiling the C++ Extension

The high-level planner relies on a compiled C++ extension for efficient incremental reachability computation.

The source code is located in:

```text
src/assembly/cpp
```

Before compilation, update the following variables in the Makefile:

- `MACC_HOME`
- `PYBIND_INCLUDE`
- `PYTHON_INCLUDE`
- `PYTHON_EXT_SUFFIX` if necessary

The extension depends on:

- pybind11
- Boost MultiArray

---

## Running the Planner

The complete planning pipeline is executed through the scripts in:

```text
src/parallelization
```

The provided scripts can be used to run benchmark experiments and analyze planner outputs. Additional usage examples will be added as the repository is updated.

---

## Benchmark Structures

The repository contains benchmark structures used in the paper.

Structure files are stored as NumPy arrays in:

```text
src/data/
```

The benchmark consists of 200 randomly generated structures containing multiple cuboid block types.

Structures can be loaded directly using:

```python
Assembly_Node.from_npy(...)
```

---

## Visualization

MeshCat is used to visualize construction sequences and generate animations.

Visualization utilities are located in:

```text
src/visualize
```

The visualization module supports:

- Construction sequence visualization
- World visualization
- MeshCat animations
- Video generation from planning results

---

## Citation

If you use this repository in your research, please cite:

```bibtex
% Citation will be added after publication.
```

---

## Authors

- Junil Min — Carnegie Mellon University
- Shambhavi Singh — University of Michigan
- Geordan Gutow — Michigan Technological University
- Smrithi Lokesh Seramalanna — University of Pennsylvania
- Bhaskar Vundurthy — Carnegie Mellon University
- Jinwoo Choi — Carnegie Mellon University
- Howie Choset — Carnegie Mellon University