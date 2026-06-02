
import os, sys, math, random, glob
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data as torch_data

from torchdrug import core, data, utils
from torchdrug.utils import comm
from torchdrug.core import Registry as R
from torchdrug.core import Configurable
# 确保可以 import gearbind
sys.path.append(os.path.abspath("."))
from gearbind import dataset as gb_dataset, model as gb_model, task as gb_task, util as gb_util, geometry, layer, residue_constants
from gearbind.dataset import bio_load_pdb, SKEMPI

import esm
import torch


esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model.eval()

def extract_esm2_representation_esm(sequence: str) -> torch.Tensor:
    batch_labels, batch_strs, batch_tokens = batch_converter([("protein1", sequence)])
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[30], return_contacts=False)
    token_reps = results["representations"][30]  # (B, L, H)

    
    token_reps = token_reps[:, 1:-1, :]      # (B, L-2, H)
    token_reps = token_reps.squeeze(0)  # (L,H)
    return token_reps


@R.register("datasets.SingleChainMutation")
class SingleChainMutation(SKEMPI):

    processed_file = None
    splits = ["single"]

    def __init__(self, path, csv_name="data.csv", verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        
        self.original_lazy = kwargs.get("lazy", False)
       

        
        pdb_files = sorted(glob.glob(os.path.join(path, "*.pdb")))
        if len(pdb_files) == 0:
            raise RuntimeError(f"No PDB files found in {path}")

        
        self.load_pdbs(pdb_files, verbose=verbose, **kwargs)

        
        csv_path = os.path.join(path, csv_name)
        if not os.path.exists(csv_path):
            raise RuntimeError(f"CSV file not found: {csv_path}")

        self.load_annotation_single(csv_path, verbose=verbose, lazy=self.original_lazy)

        
        self.num_samples = [len(self.data)]

    def load_annotation_single(self, csv_file, verbose=1, lazy=False):

        
        data_dict = {
            os.path.splitext(os.path.basename(pdb_file))[0]: (protein, pdb_file, sequence)
            for pdb_file, protein, sequence in zip(self.pdb_files, self.data, self.sequences)
        }

        
        self.data = []
        self.pdb_files = []
        self.sequences = []

        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            pdbid = str(row["pdbid"])
            wt_name = str(row["wt"])
            mut_name = str(row["mut"])
            ddG = float(row["ddg"])

            
            wt_pdb_file = os.path.join(self.path, f"{wt_name}.pdb")
            mut_pdb_file = os.path.join(self.path, f"{mut_name}.pdb")
            
            
            if not os.path.exists(wt_pdb_file) or not os.path.exists(mut_pdb_file):
                if verbose:
                    print(f"[WARN] {wt_name}.pdb or {mut_name}.pdb not found, skip.")
                continue

            
            try:
                parts = pdbid.split("_")
                _, chain, mut_code = parts[0], parts[1], parts[2]
                from_aa = mut_code[0]
                pos = int(mut_code[1:-1])
                to_aa = mut_code[-1]
            except Exception as e:
                if verbose:
                    print(f"[WARN] Fail to parse mutation from pdbid: {pdbid} ({e}), skip.")
                continue

            
            if not lazy:
                if wt_name not in data_dict or mut_name not in data_dict:
                    if verbose:
                        print(f"[WARN] {wt_name} or {mut_name} not found in PDB dict, skip.")
                    continue

                wild_type, _, wt_seq = data_dict[wt_name]
                mutant, _, mut_seq = data_dict[mut_name]
                
                if wild_type is None or mutant is None:
                    if verbose:
                        print(f"[WARN] {wt_name} or {mut_name} is None, skip.")
                    continue


                with wild_type.node():
                    
                    entity_a_res = torch.ones(wild_type.num_residue, dtype=torch.bool)
                    wild_type.entity_a = entity_a_res[wild_type.atom2residue]

                    entity_b_res = torch.zeros(wild_type.num_residue, dtype=torch.bool)
                    wild_type.entity_b = entity_b_res[wild_type.atom2residue]

                    
                    is_mut_res = torch.zeros(wild_type.num_residue, dtype=torch.bool)
                    mut_chain_id = wild_type.alphabet2id[chain]
                    is_mut_res |= (wild_type.chain_id == mut_chain_id) & (wild_type.residue_number == pos)
                    wild_type.is_mutation = is_mut_res[wild_type.atom2residue]

                
                wild_type = wild_type.subgraph(wild_type.entity_a | wild_type.entity_b)
                if hasattr(wild_type, "node_feature"):
                    with wild_type.node():
                        wild_type.node_feature = wild_type.node_feature.to_sparse()

                # --- mutant ---
                with mutant.node():
                    entity_a_res = torch.ones(mutant.num_residue, dtype=torch.bool)
                    mutant.entity_a = entity_a_res[mutant.atom2residue]

                    entity_b_res = torch.zeros(mutant.num_residue, dtype=torch.bool)
                    mutant.entity_b = entity_b_res[mutant.atom2residue]

                    is_mut_res = torch.zeros(mutant.num_residue, dtype=torch.bool)
                    mut_chain_id = mutant.alphabet2id[chain]
                    is_mut_res |= (mutant.chain_id == mut_chain_id) & (mutant.residue_number == pos)
                    mutant.is_mutation = is_mut_res[mutant.atom2residue]

                mutant = mutant.subgraph(mutant.entity_a | mutant.entity_b)
                if hasattr(mutant, "node_feature"):
                    with mutant.node():
                        mutant.node_feature = mutant.node_feature.to_sparse()
            else:
                
                wild_type = None
                mutant = None
                
                wt_seq = None
                mut_seq = None


            self.data.append((wild_type, mutant, float(ddG), [mut_code], pdbid))
            self.pdb_files.append(mut_pdb_file)
            self.sequences.append((wt_seq, mut_seq))

        if len(self.data) == 0:
            raise RuntimeError(f"No valid samples parsed from {csv_file}")
        
    def get_item(self, index):
        try:
            if getattr(self, "lazy", False):
                mutant_pdb_file = self.pdb_files[index]
                try:
                    
                    mutant, _ = bio_load_pdb(mutant_pdb_file)
                except Exception as e:
                    print(f"[WARN] Failed to load mutant PDB file {mutant_pdb_file} using bio_load_pdb: {e}")
                    
                    try:
                        mutant = data.Protein.from_pdb(mutant_pdb_file, self.kwargs)
                    except Exception as e2:
                        print(f"[WARN] Failed to load mutant PDB file {mutant_pdb_file} using from_pdb: {e2}")
                        
                        mutant = data.Protein(
                            edge_list=[[0, 0, 0]], 
                            num_node=1, 
                            num_residue=1,
                            atom_type=[0], bond_type=[0], residue_type=[0], atom2residue=[0]
                        )
                
                
                wt_pdb_file = os.path.join(
                    os.path.dirname(mutant_pdb_file),
                    os.path.basename(mutant_pdb_file).replace("mut_", "wt_")
                )
                try:
                    
                    wild_type, _ = bio_load_pdb(wt_pdb_file)
                except Exception as e:
                    print(f"[WARN] Failed to load wild type PDB file {wt_pdb_file} using bio_load_pdb: {e}")
                    
                    try:
                        wild_type = data.Protein.from_pdb(wt_pdb_file, self.kwargs)
                    except Exception as e2:
                        print(f"[WARN] Failed to load wild type PDB file {wt_pdb_file} using from_pdb: {e2}")
                        
                        wild_type = data.Protein(
                            edge_list=[[0, 0, 0]], 
                            num_node=1, 
                            num_residue=1,
                            atom_type=[0], bond_type=[0], residue_type=[0], atom2residue=[0]
                        )
            else:
                
                if self.data[index] is None:
                    raise ValueError(f"Invalid sample at index {index}")
                wild_type = self.data[index][0].clone()
                mutant = self.data[index][1].clone()
        except (ValueError, AttributeError, TypeError) as e:
            
            print(f"[WARN] Failed to process sample {index}: {e}")
           
            try:
                mutant_pdb_file = self.pdb_files[index]
                mutant = data.Protein.from_pdb(mutant_pdb_file, self.kwargs)
                wt_pdb_file = os.path.join(
                    os.path.dirname(mutant_pdb_file),
                    os.path.basename(mutant_pdb_file).replace("mut_", "wt_")
                )
                wild_type = data.Protein.from_pdb(wt_pdb_file, self.kwargs)
            except Exception as e2:
                print(f"[WARN] Failed to reload sample {index}: {e2}")
               
                wild_type = data.Protein(
                    edge_list=[[0, 0, 0]], 
                    num_node=1, 
                    num_residue=1,
                    atom_type=[0],  
                    bond_type=[0],  
                    residue_type=[0],  
                    atom2residue=[0]  
                )
                mutant = data.Protein(
                    edge_list=[[0, 0, 0]], 
                    num_node=1, 
                    num_residue=1,
                    atom_type=[0],  
                    bond_type=[0],  
                    residue_type=[0],  
                    atom2residue=[0]  
                )
        


        wt_seq, mt_seq = self.sequences[index]
        

        if wt_seq is None or mt_seq is None:
            
            def get_residue_symbol(res):
                res_id = res.item() if hasattr(res, 'item') else res
                return data.Protein.id2residue_symbol.get(res_id, 'X')
            
            wt_seq = "".join([get_residue_symbol(res) for res in wild_type.residue_type.tolist()])
            mt_seq = "".join([get_residue_symbol(res) for res in mutant.residue_type.tolist()])
        
        wt_esm = extract_esm2_representation_esm(wt_seq)
        mt_esm = extract_esm2_representation_esm(mt_seq)

       
        if wt_esm.shape[0] != wild_type.num_residue:
            print(f"[WARN] wt_esm length {wt_esm.shape[0]} != num_residue {wild_type.num_residue}")
        if mt_esm.shape[0] != mutant.num_residue:
            print(f"[WARN] mt_esm length {mt_esm.shape[0]} != num_residue {mutant.num_residue}")

      
        wt_residue_feature = F.one_hot(
            wild_type.residue_type, len(data.Protein.residue2id) + 1
        ).float()
        mt_residue_feature = F.one_hot(
            mutant.residue_type, len(data.Protein.residue2id) + 1
        ).float()

      
        wt_residue_feature_fused = torch.cat(
            [wt_residue_feature, wt_esm.to(wt_residue_feature.device)],
            dim=-1
        )

        mt_residue_feature_fused = torch.cat(
            [mt_residue_feature, mt_esm.to(mt_residue_feature.device)],
            dim=-1
        )


        wt_atom_one_hot = F.one_hot(
            wild_type.atom_name, residue_constants.atom_type_num
        ).float()
        mt_atom_one_hot = F.one_hot(
            mutant.atom_name, residue_constants.atom_type_num
        ).float()

        wt_atom_feature = torch.cat(
            [
                wt_atom_one_hot,
                wt_residue_feature_fused[wild_type.atom2residue]
            ],
            dim=-1
        )

        mt_atom_feature = torch.cat(
            [
                mt_atom_one_hot,
                mt_residue_feature_fused[mutant.atom2residue]
            ],
            dim=-1
        )


        with wild_type.node():
            wild_type.node_feature = wt_atom_feature
        with wild_type.residue():
            wild_type.residue_feature = wt_residue_feature_fused

        with mutant.node():
            mutant.node_feature = mt_atom_feature
        with mutant.residue():
            mutant.residue_feature = mt_residue_feature_fused
        

        item = {"wild_type": wild_type, "mutant": mutant}
        if self.transform:
            item = self.transform(item)
        item["ddG"] = self.data[index][2]
        return item


def main():
    
    DATA_PATH = "/fs-computility-new/UPDZ11_zhanglujia/zhangchuanxi.p/ddg/comb_final/d4126_clean"
    CSV_NAME = "data.csv"


    dataset = SingleChainMutation(DATA_PATH, csv_name=CSV_NAME, verbose=1, lazy=True)


    def random_split_dataset(ds, train_ratio=0.8, valid_ratio=0.1, seed=0):
        n = len(ds)
        indices = list(range(n))
        random.Random(seed).shuffle(indices)

        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        train_idx = indices[:n_train]
        valid_idx = indices[n_train:n_train + n_valid]
        test_idx = indices[n_train + n_valid:]

        train_set = torch_data.Subset(ds, train_idx)
        valid_set = torch_data.Subset(ds, valid_idx)
        test_set = torch_data.Subset(ds, test_idx)
        return train_set, valid_set, test_set

    train_set, valid_set, test_set = random_split_dataset(dataset, train_ratio=0.8, valid_ratio=0.1, seed=42)


    USE_KNN_MUT_SITE = False


    if USE_KNN_MUT_SITE:
        node_layers_cfg = [
            {
                "class": "KNNMutationSite",
                "k": 64,
            }
        ]
    else:
        node_layers_cfg = []

    graph_construction_cfg = {
        "class": "GraphConstruction",
        "node_layers": node_layers_cfg,
        "edge_layers": [
            {"class": "SequentialEdge", "max_distance": 2},
            {"class": "SpatialEdge", "radius": 10.0, "max_distance": 5},
            {"class": "KNNEdge", "k": 10, "max_distance": 5},
        ],
        "edge_feature": "gearnet",
    }


    task_cfg = {
        "class": "BindingAffinityChange",
        "model": {
            "class": "BindModel",
            "num_mlp_layer": 2,
            "model": {
                "class": "GearBind",
                "input_dim": 698,  
                "hidden_dims": [128, 128, 128, 128],
                "batch_norm": True,
                "short_cut": True,
                "concat_hidden": True,
                "num_relation": 7,
                "edge_input_dim": 59,
                "num_angle_bin": 8,
                },
        },
        "graph_construction_model": graph_construction_cfg,
        "normalization": True,
        "task": {"ddG": 1.0},
        "criterion": "mse",
        "metric": ["mae", "rmse", "spearmanr", "pearsonr"],
    }

    bind_task = core.Configurable.load_config_dict(task_cfg)
    print(bind_task)


    pretrained_model_path = "/fs-computility-new/UPDZ11_zhanglujia/zhangchuanxi.p/ddg/comb_final/checkpoints_smega/scratch_smega_ddg_best_full_task.pth"
    if os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from {pretrained_model_path}...")
        ckpt = torch.load(pretrained_model_path, map_location="cpu")
        bind_task.load_state_dict(ckpt["model"], strict=False)
        print("Pretrained model loaded successfully!")
    else:
        print(f"Warning: Pretrained model not found at {pretrained_model_path}")

    for param in esm_model.parameters():
        param.requires_grad = False


    optimizer = torch.optim.Adam(bind_task.parameters(), lr=1e-4)


    gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    BATCH_SIZE = 2

    solver = core.Engine(
        bind_task,
        train_set,
        valid_set,
        test_set,
        optimizer,
        gpus=gpus,
        batch_size=BATCH_SIZE,
        num_worker=4,
        log_interval=10
    )

 
    import swanlab
    run = swanlab.init(logdir="logs_fine_tune", log_level="INFO")


    NUM_EPOCH = 50
    os.makedirs("checkpoints_fine_tune", exist_ok=True)


    TAG = f"ddg_fine_tune"
    best_metric = float("inf")
    best_epoch = -1

    best_full_path = f"checkpoints/{TAG}_best_full_task.pth"

    for epoch in range(NUM_EPOCH):
        print(f"\n===== Epoch {epoch+1}/{NUM_EPOCH} =====")
        solver.train(1)
        train_loss = np.mean(solver.meter.records['mean squared error'])


        metric_dict = solver.evaluate("valid")
        print("valid metrics:", metric_dict)

        cur_metric = metric_dict["mean absolute error [ddG]"].item()
        

        mae = metric_dict["mean absolute error [ddG]"].item()
        rmse = metric_dict["root mean squared error [ddG]"].item()
        spearmanr = metric_dict["spearmanr [ddG]"].item()
        pearsonr = metric_dict["pearsonr [ddG]"].item()


        swanlab.log({
            "train_loss": train_loss,
            "mae": mae,
            "rmse": rmse,
            "spearmanr": spearmanr,
            "pearsonr": pearsonr
        })

        if cur_metric < best_metric:
            best_metric = cur_metric
            best_epoch = epoch + 1

           
            torch.save(
                {"model": bind_task.state_dict()},
                best_full_path,
            )

            print(f"  >> New best! mae = {best_metric:.4f} @ epoch {best_epoch}, saved.")

    print(f"\nTraining finished. Best valid mae = {best_metric:.4f} at epoch {best_epoch}")
    print("Best full-task ckpt:", best_full_path)


    print("\n=== Evaluate on TEST set with best model loaded ===")
    bind_task_best = Configurable.load_config_dict(task_cfg)
    ckpt_best = torch.load(best_full_path, map_location="cpu")
    bind_task_best.load_state_dict(ckpt_best["model"], strict=False)

    optimizer_dummy = torch.optim.Adam(bind_task_best.parameters(), lr=1e-4)
    solver_best = core.Engine(
        bind_task_best,
        train_set,
        valid_set,
        test_set,
        optimizer_dummy,
        gpus=gpus,
        batch_size=BATCH_SIZE,
    )

    solver_best.evaluate("test")

if __name__ == "__main__":
    main()
