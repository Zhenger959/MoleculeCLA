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
| Biological | docking\_score (docking score)      | Docking score                                 | Binding affinity              |
|          | glide\_emodel (emodel)         | Model energy                                  | Binding affinity              |

## Getting Started
### Prerequisites
MoleculeCLA data and model latent representations used in our paper can download from [HuggingFace](https://huggingface.co/datasets/anonymousxxx/MoleculeCLA).

### Data Format

- The 'data.csv' file contains information on scaffold splitting for training, testing, and validation sets, along with the SMILES representations of molecules and their corresponding molecular IDs for identification.

- The 'labels/*.csv' file contains data on molecular properties derived from binding analysis, along with their corresponding molecule IDs, Each file name corresponds to a specific protein target name.

- The 'docking_id_idx_map.json' file provides the mapping from molecule IDs to the index in the 'diversity_molecule_set.pkl' file.

- The 'diversity_molecule_set.pkl' file contains the 3D coordinates of molecules, necessary for 3D-based molecular representation learning methods. The elements in the pickle file are structured as follows:

```
{'ID': 'SC98-0239', 'coordinates': [array([[-4.11016704, -3.37916002, -0.88502367],
       [-2.81365324, -2.85697084, -1.50493104],
       [-2.91199571, -2.78934427, -3.02901557],
       [-2.42701714, -1.56525244, -0.94587911],
       [-1.23215803, -1.21899151, -0.39144257],
       [-1.31749551,  0.11476489, -0.05530111],
       [-0.20480789,  0.877684  ,  0.57378243],
       [ 1.08304448,  0.1660573 ,  0.35183821],
       [ 2.14510035,  0.84228324,  1.11256913],
       [ 3.49157168,  0.39465333,  0.64371576],
       [ 4.0093678 ,  0.45211187, -0.6360267 ],
       [ 5.25977664, -0.07824883, -0.57110278],
       [ 6.18873586, -0.24852661, -1.66094418],
       [ 5.60951998, -0.47724683,  0.66438966],
       [ 4.5424841 , -0.17884145,  1.41489069],
       [ 4.55558251, -0.47107272,  2.8709214 ],
       [ 0.97871877, -1.27411476,  0.70842627],
       [-0.0145135 , -2.04472413, -0.20531951],
       [-3.30448069, -0.53833909, -0.95941011],
       [-2.6346263 ,  0.4881592 , -0.40801827],
       [-3.27106052,  1.91973232, -0.18176253],
       [-4.50778755,  2.25235937, -0.51859896],
       [-4.63240329,  3.56087824, -0.13563034],
       [-3.46126275,  3.99574545,  0.43161618],
       [-2.99734689,  5.26214595,  1.01808915],
       [-2.57726543,  2.95578027,  0.41052655]])], 'atoms': ['C', 'C', 'C', 'N', 'C', 'C', 'C', 'N', 'C', 'C', 'C', 'N', 'C', 'N', 'C', 'C', 'C', 'C', 'N', 'C', 'C', 'N', 'C', 'C', 'C', 'O'], 'smi': 'CC(C)n(c1c2CN(Cc3cn(C)nc3C)CC1)nc2-c1ncc(C)o1', 'mol': <rdkit.Chem.rdchem.Mol object at 0x7fe70e732f40>}
```

### Environment
```
conda create -n py3.9 python=3.9
conda activate py3.9
pip install -r requirements.txt
```


### Linear Probe
Using the model latent representations or descriptors, a linear regression model is trained to predict the property label.
```
python src/linear_probe.py \
    -m  Linear_regression \
    -s scaffold \
    -i data/model_feature/descriptors_3D.pkl \
    -f data/data.csv \
    -l data/labels \
    -o res
```

### Multi-Layer Perceptron
We use the following settings in our Multi-Layer Perceptron experiments. Each experiment is conducted on an NVIDIA A100-PCIE-40GB GPU and takes approximately half an hour to converge.

You can directly run the code by executing the shell command `bash scripts\mlp.sh`.
```
LR=1e-4
BATCH_SIZE=128
MAX_EPOCHS=50
DROPOUT_RATE=0.0
WARMUP=0.02

LOG_DIR='logs'
FEATURE_NAME="mols_unimol.pkl"
TASK_NAME="docking_score"
TARGET="ADRB2"
EMBED_PATH="data/model_feature"

python src/mlp_main.py \
    experiment=moleculecla_mlp \
    tags=\["${FEATURE_NAME}","${TASK_NAME}"\] \
    task_name=${FEATURE_NAME}_${TARGET}_${TASK_NAME} \
    data.task_idx=${TASK_IDX} \
    data.train.dataset.target=${TARGET} \
    data.val.dataset.target=${TARGET} \
    data.test.dataset.target=${TARGET} \
    data.train.dataset.embed_path=${EMBED_PATH} \
    data.val.dataset.embed_path=${EMBED_PATH} \
    data.test.dataset.embed_path=${EMBED_PATH} \
    model.optimizer.lr=${LR} \
    data.batch_size=${BATCH_SIZE} \
    trainer.max_epochs=${MAX_EPOCHS} \
    model.model.dropout_rate=${DROPOUT_RATE} \
    model.lr_scheduler.warmup=${WARMUP}
```

### Fine-tuning
We loaded the parameter values from the offical pre-trained models checkpoints, and fine-tune them on five representative tasks (hbond, ecoul, esite, docking and emodel). In our paper, we train a separate MLP model for each protein target and simultaneously predict these tasks. The detailed hyperparameters used in each pre-training method are listed in the table below. Specifically, UniMAP employs grid search to select the optimal learning rate and weight decay for each task. The search ranges are provided in the corresponding positions of the table.


| Methods | Optimizer | Learning rate     | Weight decay      | Epoch | Batch size |
|---------|-----------|-------------------|-------------------|-------|------------|
| Uni-Mol | Adam      | 4.00E-04          | 0                 | 50    | 128        |
| UniMAP  | Adam      | (1e-6, 1e-4)      | (1e-7, 1e-3)      | 50    | 64         |
| Frad    | Adam      | 1.00E-04          | 0                 | 100   | 32         |
| SliDe   | Adam      | 1.00E-04          | 0                 | 100   | 32         |


## License
Our dataset is available under the MIT license. The evaluation code is hosted by the GitHub organization and uses the MIT license.

<!-- ## Contributing



## Contact -->
