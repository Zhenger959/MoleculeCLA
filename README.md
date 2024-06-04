# MoleculeCLA
## Overview
We present MoleculeCLA: a large-scale dataset consisting of approximately 140,000 small molecules derived from computational ligand-target binding analysis, providing nine properties that cover chemical, physical, and biological aspects.

| Aspect   | Glide Property (Abbreviation) | Description                                   | Molecular Characteristics     |
|----------|--------------------------------|-----------------------------------------------|------------------------------|
| Chemical | glide\_lipo (lipo)             | Hydrophobicity                                | Atom type, number            |
|          | glide\_hbond (hbond)           | Hydrogen bond formation propensity            | Atom type, number            |
| Physical | glide\_evdw (evdw)             | Van der Waals energy                          | Size and polarizability      |
|          | glide\_ecoul (ecoul)           | Coulomb energy                                | Ionic state                  |
|          | glide\_esite (esite)           | Polar thermodynamic contribution              | Polarity                     |
|          | glide\_erotb (erotb)           | Rotatable bond constraint energy              | Rotational flexibility       |
|          | glide\_einternal (einternal)   | Internal torsional energy                     | Rotational flexibility       |
| Biological | docking\_score (docking)      | Docking score                                 | Binding affinity              |
|          | glide\_emodel (emodel)         | Model energy                                  | Binding affinity              |

## Getting Started

### Prerequisites
MoleculeCLA can download from https://huggingface.co/api/datasets/shikun001/MoleculeCLA

Model latent representations and descriptors can download from.

### Installation
```
conda create -n py3.9 python=3.9
conda activate py3.9
pip install -r requirements.txt
```

## Usage
We provide the linear probe code and MLP code，ensuring that all results are easily reproducible.
### Linear Probe
```
python src/linear_probe.py \
    -m  Linear_regression \
    -s scaffold \
    -i data/model_feature/descriptors_3D.pkl \
    -f data/data.csv \
    -l data/labels \
    -o res
```

### MLP
```
bash scripts\mlp.sh
```

## License
Our dataset is available under the CC-BY license. The evaluation code is hosted by the GitHub organization and uses the MIT license.

<!-- ## Contributing



## Contact -->