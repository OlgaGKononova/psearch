# PSearch - 3D ligand-based pharmacophore modeling

PSearch is a tool to generate 3D ligand-based pharmacophore models and perform virtual screening with them.

## Installation

```bash
pip install psearch
pip install -U git+https://github.com/meddwl/psearch.git
```

## Dependency
 
`python >= 3.4`  
`rdkit >= 2017.09`  
`networkx >= 2`  
`pmapper >= 0.4.1`  

## Example
The demonstration of the tool is carried out on the example of the target CHEMBL5719. A shortened sample of 233 structures is proposed.

### Creation of ligand-based pharmacophore models
It is recommended to create an empty dir which would be your `$PROJECT_DIR` and copy an input file to that location.
In our case, PROJECT_DIR=example/test.
There are two steps of pharmacophore model generation.

1. Generation of a database with precomputed conformers and pharmacophores. 

```python
gen_db -i cdk8.smi -o dbs/cdk8.dat -c 4 -v
```
`-i` - path to the input SMILES file  
`-o` - path to database (should have extension .dat)  
`-c` - number of CPUs to use  
`-v` - print progress to STDERR  
There are other arguments which one can tune. Invoke script with `-h` key to get full information.  
Generating the database on 4 cores will take up to 15 minutes  

The script generates stereoisomers and conformers, creates the database of compounds with labeled pharmacophore features.  

The script takes as input a tab-separated SMILES file containing `SMILES`, `compound id`, `activity` columns. 
The third column should contain a number `1` or `0`. If there are bad structures in the input smi-file they 
will be omitted and a new smi-file will be created without these structures. The original input file 
will be backed up with another name (`#cdk8.smi.1#`).

2. Model building.  

```python
psearch -p my_models/created_pharmacophores/ -i cdk8.smi -d dbs/cdk8.dat -c 4
```
`-p` - path to the models directory where training set files, pharmacophore models, screening results and external model statistics will be stored  
`-i` - path to the input SMILES file  
`-d` - path to the database generated on the previous step  
`-c`- number of CPUs to use  

Other arguments are available at the command line.

This will create several folders within the `my_models` folder. The `models` folder contains the generated pharmacophore models. 
File names use the following naming convention: `<database_name>.t<train_set_number>_f<number_of_features>_p<sequentional_model_number>.xyz`.
The `raw_screen` folder contains results of screening of the compounds in database. External statistics for molecules of the test set will be calculated.
The `trainset` folder contains training sets. `external_statistics.txt` is a file with validation statistics 
(molecules which were not used to train particular models are used as corresponding test sets).

You will receive a progress notification:
```
train set t2: 2 models (112.368s)
train set t1: 14 models (134.545s)
train set t3: 1 models (155.38s)
train set centroids: 4 models (52.758s)
train set t0: 2 models (287.577s)
train set t4: 1 models (265.789s)
230 molecules screened 00:00:12
external_statistics.txt: (0.587s)
```

### Virtual screening of a chemical database using pharmacophore models 

1. Database creation using the same procedure as described above.   
*we skip this step here and will use the same database used for model training.

2. Virtual screening.
  
```python
screen_db -d dbs/cdk8.dat -q my_models/created_pharmacophores/models/ -o my_models/vs/ -c 4 -v
```
`-d` - input generated database  
`-q` - pharmacophore model or models or a directory with models. If a directory would be specified all pma- and xyz-files will be recognized as pharmacophores and will be used for screening  
`-o` - path to an output directory if multiple models were supplied for screening or a path to a text/sdf file  
`-c`- number of CPUs to use  
`-v` - print progress to STDERR  

If sdf output is desired a user should add `-z` argument which will force output format to be sdf.

3. Calculating probability of the activity of molecules towards the protein and rank molecules.  

```python
prediction -s my_models/vs/models -p my_models/created_pharmacophores/external_statistics.txt -f max -o my_models/results.txt
```
`-s` - path to the virtual screening result  
`-p` - file with the calculated precision of pharmacophore models  
`-f` - one of the two schemes (max and mean) of probability calculation for consensus prediction based on the individual precision of pharmacophore models 
`-o` - output text file where will be saved the prediction


### Profiling of molecules using multiple pharmacophores

1. Database creation using the same procedure as described above. 

The following protocol can be used for profiling of molecules. By default, multiprofiling is performed on 
psearch ligand-based pharmacophore models which were cheated using data from ChEMBL. Additional information about 
the psearch pharmacophore models can be found in the pharmacophores folder.

```python
gen_db -i mols_for_profiling.smi -o dbs/mols_for_profiling.dat -c 4 -v
```

2. Virtual screening.
  
```python
screen_db -d dbs/mols_for_profiling.dat -o multiprofiling/vs/ -c 4 -v
```

3. Calculating probability of the activity of molecules towards the protein and rank molecules.
The scheme of how the probability is calculated is described in the [article](https://doi.org/10.3390/molecules25020385) below

```python
prediction -s multiprofiling/vs/ -o multiprofiling/result_multiprofiling.txt
```
`-s` - path to the virtual screening result  
`-o` - output text file where will be saved the prediction 

## Documentation

All scripts have `-h` argument to retrieve descriptions of all available options and arguments.

## Authors
Alina Kutlushina, Pavel Polishchuk

## Citation
Ligand-Based Pharmacophore Modeling Using Novel 3D Pharmacophore Signatures  
Alina Kutlushina, Aigul Khakimova, Timur Madzhidov, Pavel Polishchuk  
*Molecules* **2018**, 23(12), 3094  
https://doi.org/10.3390/molecules23123094

Probabilistic Approach for Virtual Screening Based on Multiple Pharmacophores  
Timur Madzhidov, Assima Rakhimbekova, Alina Kutlushuna, Pavel Polishchuk  
*Molecules* **2020**, 25(2), 385  
https://doi.org/10.3390/molecules25020385  

## License
BSD-3 clause
