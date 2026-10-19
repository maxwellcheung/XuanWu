IsPETase S185H Single-Mutation Prediction Example

## Input Description

`input/data.csv` contains only one record:

```csv
pdbid,wt,mut,substitution,mutation,ddg
5XH3_A_S185H,wt_5XH3_A_S185H,mut_5XH3_A_S185H,S185H,A185H,0.0
Field meanings:

Field	Meaning
pdbid	Combination identifier for the structure, chain, and mutation
wt	Wild-type PDB filename (without .pdb)
mut	Mutant PDB filename (without .pdb)
substitution	Biological mutation name, i.e., Ser185→His
mutation	Model locator field: A indicates chain A, 185 indicates the residue number, and H indicates the residue after mutation
ddg	Placeholder label required for data loading; it does not participate in prediction during inference and is fixed at 0.0
Please note that the first A in mutation=A185H is a chain identifier; it does not mean that the wild-type residue is alanine. The wild-type and mutant PDBs have been verified to have SER and HIS, respectively, at position 185 of chain A.

Running Prediction Separately
bash
python IsPETase_S185H_example/predict_s185h.py \
  --device auto
By default, the output is written to output/prediction_output.csv.

Output Example
csv
pdbid,substitution,model_locator,ddg_pred_kcal_mol,execution_device
5XH3_A_S185H,S185H,A185H,-0.28827786445617676,cuda:0
Here, ddg_pred_kcal_mol is the predicted ΔΔG (kcal/mol) directly output by the model; the script does not flip the sign. execution_device records the actual inference device. The sign should be interpreted according to the ΔΔG definition stated in the paper.

