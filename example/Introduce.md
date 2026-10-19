# IsPETase S185H 单突变预测示例


```

## 输入说明

`input/data.csv` 仅包含一条记录：

```csv
pdbid,wt,mut,substitution,mutation,ddg
5XH3_A_S185H,wt_5XH3_A_S185H,mut_5XH3_A_S185H,S185H,A185H,0.0
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `pdbid` | 结构、链和突变的组合标识符 |
| `wt` | 野生型 PDB 文件名（不含 `.pdb`） |
| `mut` | 突变型 PDB 文件名（不含 `.pdb`） |
| `substitution` | 生物学突变名称，即 Ser185→His |
| `mutation` | 模型定位字段：`A` 表示链 A，`185` 表示残基编号，`H` 表示突变后残基 |
| `ddg` | 数据加载所需的占位标签；推理时不参与预测，固定为 `0.0` |

请注意，`mutation=A185H` 中的第一个 `A` 是**链标识符**，并不表示野生型残基为丙氨酸。野生型和突变型 PDB 已核验为链 A 第 185 位分别是 `SER` 和 `HIS`。


## 单独执行预测

```bash
python IsPETase_S185H_example/predict_s185h.py \
  --device auto
```

默认输出至 `output/prediction_output.csv`。

## 输出示例

```csv
pdbid,substitution,model_locator,ddg_pred_kcal_mol,execution_device
5XH3_A_S185H,S185H,A185H,-0.28827786445617676,cuda:0
```

其中 `ddg_pred_kcal_mol` 是模型直接输出的预测 ΔΔG（kcal/mol），脚本未对符号进行翻转；`execution_device` 记录实际推理设备。正负号应按照论文中声明的 ΔΔG 定义解释。

