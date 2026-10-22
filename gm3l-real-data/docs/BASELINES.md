# 对照方法与外部接口

本包只整理可审核的原始调用方式，不附带第三方整套源码、作者数据、历史拟合模型或结果。没有访问真实数据，没有重新跑原实验，不能据此宣称完整论文数值复现。此表中的历史来源路径是审计定位信息，不是运行时依赖。

## 已实现的轻量适配层

`fit_baseline(method, X_train, X_test, K, seed, options=None)` 不接受 `y` 或图输入，返回训练/测试硬标签、测试连续分数、列对应的原生簇 ID、分数元数据、诊断及少量拟合数组。所有拟合只使用训练特征；测试特征不改变中心、训练划分或配置选择。

| 名称 | 保留的原调用 | 分数与限制 |
| --- | --- | --- |
| `FINCH` | 用户安装 `finch-clust` 后调用 `finch.FINCH(..., req_clust=K, random_state=seed)` | 原生训练层次标签；测试为到冻结训练簇均值的最近余弦距离扩展；不强迫原生结果恰好 K 簇 |
| `CLASSIX` | 用户安装 `classixclustering` 后调用 `classix.CLASSIX(...).fit(X_train)` 与原生 `predict` | 默认半径 0.5，忽略 K；保留变簇数；在训练拟合预处理空间取每簇原生 aggregation starting points 的最近距离 |
| `CLASSIX_known_K` | 同一原生 CLASSIX，加原适配层的训练几何半径策略 | `geomspace(0.02,1,25)`；先最小化原生簇数与 K 的差，再最大化训练 silhouette，再选小半径；不重写训练标签 |

也可用 `CLASSIX` 配合 `options={"radius_selection":"known_K"}`。所有具名预设只允许重述参数，不允许用 `options` 悄悄更改主实验参数。CLASSIX 原适配层会保存和恢复 NumPy/Python 的全局随机状态；因此应串行调用或使用独立进程，不适合并发线程。

FINCH 原固定参数为 cosine 距离、`ensure_early_exit=True`、`ann_threshold=20000`、`faiss_threshold=5000000`、`faiss_use_gpu=False`、`return_meta_labels=False`、`verbose=False`。大样本近邻后端依赖由用户安装的原包负责。

CLASSIX 原固定参数为 `sorting='pca', metric='euclidean', radius=0.5, minPts=1, group_merging='distance', mergeScale=1.5, post_alloc=True, mergeTinyGroups=True, verbose=0, short_log_form=True`。打分依赖该包的原生 starting-point 属性；若版本接口变化会报错，不静默改用另一种算法。

来源：`Xonly_realdata_comparison_20261009/adapters/classical.py` 和 `auc_adapters/classical_scores.py`。原源码没有提供这些三方包的可核实网址，因此此处不补猜 URL。

full/tied GMM 属于本包单独的原自写 EM 路径，绝不经此接口改成 `sklearn.mixture.GaussianMixture`。tied 来源为 `realdata_gmm_tied_fixed50_20261012/fit_tied.py`：只将原 GMM 的协方差 M 步改为共同加权协方差，原先无正则化。

## 外部方法清单与边界

下表列的是本次源代码核对覆盖的方法。标记“外部”的方法本体没有打包；用户必须自行安装原包/作者代码。DEC 和 DipEncoder 已有本包提供的实际训练/评分 CSV 桥接；其他未注明已有桥接的方法仍需提供符合下节协议的脚本。原历史适配层包含绝对路径、固定形状、旧 `config.json` 和旧 `.npy` 输入，不能直接当作便携脚本调用。所有网址均已出现在核对过的源代码中，未联网获取或验证远端可用性。

| 名称 | 源码中的来源 | 执行依赖 / 原有差异 | 本包状态与未验证项 |
| --- | --- | --- | --- |
| DEC (2016) | `Xonly.../adapters/dec_dip.py`；未见 URL | Python、PyTorch、`clustpy==0.0.3`、threadpoolctl；保留 DEC Student-t q，不用库额外最终 KMeans 标签 | 已提供 `external_adapters/run_clustpy.py --method DEC` 实际调用；未运行深度训练，不能宣称数值一致 |
| DipEncoder (2022) | 同上及 `auc_adapters/dip_scores.py`；未见 URL | 同上；原生固定阈值预测，投票除以 K−1 为排序分数，不是类概率 | 已提供同一脚本 `--method DipEncoder` 实际调用；保留原生源码 SHA 门禁，深度训练未验证 |
| SpectralNet / SpectralNet2018 | [源码记录的作者仓库](https://github.com/shaham-lab/SpectralNet)，commit `aefe9e84b016c592fce3c94759fdc604c930f40d` | PyTorch、scikit-learn、tqdm、作者 spectralnet；原适配层使用训练谱映射后再训练 KMeans，谱后端 KMeans 的 `n_init=10` | 外部；不能调用会对测试重新 fit KMeans 的上游 predict 路径 |
| GCEALS / G-CEALS2025 | [源码记录的作者仓库](https://github.com/mdsamad001/G-CEALS---Deep-Clustering-for-Tabular-Data)，commit `1236d1b82e663119ed17fb35554e2c8d5d19f457` | PyTorch、scikit-learn、作者源码；旧适配层挑选声明以避开作者 CLI/下载/标签读取 | 外部；没有复制作者实现或启动下载 |
| TELL2022 | `realdata_tell_standardfc_fixed50_20261012/adapter.py`；commit `cc01da29f9ed498a41dca197d5b112dbc53747e6`；未见 URL | PyTorch、作者 `Net`；原适配维度10/25簇、训练 minmax、作者宽度500/500/2000、3001 epochs、5 starts，按训练损失选模型 | 外部；不可声称任意维度泛化为原固定实验，缓存逻辑需用户便携化 |
| attSBM2019 | `XA_realdata_comparison_20261008_figure3/adapters/run_graph_models.R` 与旧 `run_attsbm.R`；未见 URL | R、Rcpp/编译器、RSpectra、jsonlite、digest、原 patched helper；原训练改为已知 K 初始化，含预设数值保护；测试 joint mean-field 为新增扩展 | 外部；原训练与扩展均未运行，不声称纯作者原版 |
| CSBM_IRMAP2022 | 同上与旧 `run_csbm_map.R`；未见 URL | R、ClusterR、RSpectra、Rcpp/编译器、jsonlite、digest；15轮训练、训练内估计 sigma；测试 joint ICM 为新增扩展 | 外部；不能把归一化 decision confidence 说成校准后验 |
| NSBM2022 | 同上与旧 `NSBM2022_core.R` | R、Rcpp/编译器、jsonlite、digest；旧脚本明确是独立 K 类实现，原作者复现源码仅二分类 | 外部；不能标成作者原生多分类实现 |
| BCDC2025 | `run_bcdc_joint.R`；源码下载器给出 [作者仓库 Git blob API 前缀](https://api.github.com/repos/aaamini/bcdc/git/blobs/)，commit `e22fb2e27e33d795947f259b5d0cb1c7ee500021` | R、Matrix、bcdc、Rcpp/编译器、jsonlite、digest；unknown K；1500 Gibbs迭代，测试 cut-posterior/Dahl/novel-cluster 为披露的扩展 | 外部；有限 MCMC预算不等于收敛，负新簇ID与novel mass列不能静默改映射 |
| CLSBM2025 | 旧 `clsbm_adapter.py` 引用 [arXiv版本](https://arxiv.org/pdf/2501.11139v3)；最新为 `run_clsbm_transductive.py` | NumPy、SciPy、scikit-learn、threadpoolctl；全体未标记节点的谱表示、仅训练行拟合中心，内部 KMeans `n_init=30` | 外部；明确 transductive exception，不得混称训练-only表示 |
| SSC_IPA2022 | [作者仓库](https://github.com/danielgribel/SSC-IPA) 与 [原脚本所引论文](https://arxiv.org/pdf/2104.02146) | Python、NumPy、SciPy、threadpoolctl、Julia及作者项目；原作者无 held-out predictor，最新 joint ICM 是冻结训练参数后的自定义扩展 | 外部；未运行 Julia，未验证作者依赖版本/求解一致性 |
| CASC2017 | [作者仓库](https://github.com/norbertbin/rCASC)，源码记录 DOI `10.1093/biomet/asx008` | R、Matrix、irlba、jsonlite、作者 casc.R 与 solver compatibility层；原 full-graph 非行归一化谱适配与后来的 train-only 行归一化/Nyström分支有区别 | 外部；必须在方法名/诊断中区分版本，不能合并成同一复现结果 |

其他未在此次核对范围内定位到的旧模型不推断、不补实现。GM3L/SMLE/FINAL/GMM 由本包主实现及相应文档说明。

补充：CLSBM 的实际便携桥接现已提供为 `external_adapters/run_clsbm.py`，无须用户重写，见 [CLSBM 的输入及方程说明](CLSBM.md)。下节提供 DEC/DipEncoder 的实际桥接，其余表中标“外部”的方法仍需要相应上游环境与协议适配。

### DEC / DipEncoder 可执行桥接

`external_adapters/run_clustpy.py` 已移植原本地适配层的实际 estimator 构造、`fit(X_train, y=None)` 和评分逻辑，不要求用户重写 bridge，也不携带 ClustPy/PyTorch 本体。`--help` 只依赖 Python 标准库，能在未安装可选依赖时查看接口。

```text
python external_adapters/run_clustpy.py --method DEC \
  --input MODEL_ONLY_CSV_DIR --output NEW_OUTPUT_DIR --seed 42 --K 25
python external_adapters/run_clustpy.py --method DipEncoder \
  --input MODEL_ONLY_CSV_DIR --output NEW_OUTPUT_DIR --seed 42 --K 25
```

统一外部命令配置可写为：

```json
{
  "command": ["python", "external_adapters/run_clustpy.py", "--method", "DEC",
              "--input", "{input}", "--output", "{output}", "--seed", "{seed}", "--K", "25"],
  "cwd": "/absolute/path/to/this/package",
  "timeout": 3600,
  "transductive": false
}
```

DipEncoder 仅需将 `--method` 后的参数改为 `DipEncoder`。K 是用户明确给定的先验，不从真实类别推导。脚本只打开 `X_train.csv` 和 `X_test.csv`，不会打开同一模型目录中的图文件；发现标签、配置、结果等非协议文件会拒绝。缺少精确所需 ClustPy 版本或导入依赖会写 `missing_dependency`，不会改用其他方法。

固定参数与原 source defaults 一致：100 个预训练 epoch、150 个聚类 epoch；Adam，学习率分别 `1e-3`/`1e-4`；编码器 `[p,64,64,10]`、反向解码器；CPU、1线程；clustering loss weight 1；DEC batch size 256 / alpha 1；DipEncoder batch size 0（原库按 `25*K` 解释）、`ssl_loss_weight=None`、`max_cluster_size_diff_factor=3`。本桥接没有开放静默更改上述配置的选项。

DEC 使用 `dec_cluster_centers_` 和原生 `_dec_predict` 输出 Student-t q，核对训练 argmax 等于 `dec_labels_`，不采用库额外最终 KMeans 的 `labels_`。DipEncoder 保留原 `model.labels_` / `model.predict`，并按原分对顺序、投影、`<`/`>=` 阈值规则计票，除以 K−1；核对 argmax 与原生预测一致且评分前后网络状态不变。Dip 分数行和是 `K/2`，不是1，不能标成概率。Dip 原生模块必须满足原 supplement 的 SHA-256 `6e633c5c28e1cde6ee9d3519664b76bcee5d3c956dce1cd54be00cf9d7703500`，即使包版本相同但源码变化也明确失败。

输出包含统一协议五个文件，另含 `train_scores.csv` 和纯数组 `model.npz`。评估元数据显式使用 hard training Hungarian 并声明原生 `score_type`；不拿测试标签做映射。原 supplement 需要与历史预测对比的检查在这个无历史数据包中无法进行，因此 diagnostics 明确记录 `original_saved_predictions_compared=false`、`numerical_reproduction_verified=false`。只观察训练进度的历史 monkeypatch hook 没有移植，原训练步骤/损失/优化器/停机规则不替换；历史 loss trace 不冒称可用。固定 epoch 预算完成后状态为 `completed`，仅表示运行结束；`convergence.converged=null` 明确未证明优化收敛。DipEncoder 的投票比例每行和为 K/2，不是类别后验概率；保留原生分数与辅助 ACC，但报告 `mse_unavailable`，不计算 MSE_zeta，也不能据此复现论文后验 MSE 面板。

## 无标签的外部进程协议

Python API：

```python
run_external(command, inputs, output_dir, seed,
             timeout=3600, cwd=None, transductive=False)
```

`command` 是参数列表，不是 shell 字符串，内部始终 `shell=False`。支持 `{input}`、`{output}`、`{seed}`；必须出现前两项。K、训练预算、作者源码路径等在用户自己的命令参数或桥接脚本中明确配置。命令为空、可执行文件不存在或明确缺包时报告 `missing_dependency`，不会回退成别的模型。

`inputs` 必须只含模型数组。必需 `X_train`、`X_test`；可选 `A_train` 与 `A_test_train` 必须成对；只有显式 `transductive=True` 才允许 `A_test_test`。所有图块必须是二值矩阵，两个方阵须对称、对角为0，尺寸与特征行数一致。外部协议目前要求训练和测试均至少一行。

运行时另建临时目录，只写以下存在的文件，CSV 无表头：

```text
X_train.csv          n_train × p
X_test.csv           n_test × p
A_train.csv          n_train × n_train           # optional
A_test_train.csv     n_test × n_train            # optional
A_test_test.csv      n_test × n_test             # transductive only
```

不会写标签、完整原数据、评估划分文件或 `config.json`。传入 `y` / `y_train` 等额外键立即拒绝，不是静默忽略。这个边界限制导出文件，但外部脚本是用户信任的代码，不是 OS 沙箱；仅靠统一进程调用也不能证明外部脚本真的按训练/测试阶段隔离访问。若需要完整图表示/联合测试图的方法，应单独声明 transductive 并报告其信息可见性。

外部脚本必须在全新输出目录写：

```text
train_labels.csv     n_train 行，整数原生簇 ID
test_labels.csv      n_test 行，整数原生簇 ID
test_scores.csv      n_test × C，有限连续分数
score_columns.json   分数列对应关系与评估声明
diagnostics.json     原模型状态、依赖/版本/来源、信息访问、收敛等声明
```

`score_columns.json` 例（负距离分数）：

```json
{
  "cluster_ids": [0, 1],
  "score_type": "decision",
  "alignment": "hard",
  "mapping": "hungarian",
  "score_aggregation": "max"
}
```

最低要求是 `cluster_ids` 和显式 `score_type`。`cluster_ids` 唯一且按列排序；不是把真实类别 ID 填进去。`alignment='soft'` 还须提供 `train_scores.csv`。原 X-only supplement 使用训练标签 Hungarian 映射；未知 K 若改用 `mapping='majority'`，须公开其为单独评估政策。后验多对一映射在归一化 log probability 上用 logsumexp 合并，等价于概率求和。原生 `score_aggregation` 仅保留为来源元数据，不是非后验方法报告 `mse_unavailable` 的必需项。后验 MSE 仅接受 probability、log_probability 或 logits；负距离、投票及其他非后验分数保留导出，但不伪造概率。测试集缺少真实类不会使 MSE 无效；未映射类别的后验 MSE 不可用，除非显式训练先验政策已解决缺失。

`diagnostics.json` 应至少含 `completed` 和 `status`，不要把“迭代已结束”写成“已证明收敛”。适配层会保留模型标签与分数，不用 argmax 替换硬标签，不伪造概率；现有非空输出目录拒绝覆盖。错误分别报告 `missing_dependency`、`failed`、`timeout`、`invalid_output`；未知运行问题不会假装模型已成功包含。

命令配置示例（路径需用户替换，此处不是已安装的方法）：

```json
{
  "command": ["python", "/path/to/user_bridge.py", "--input", "{input}",
              "--output", "{output}", "--seed", "{seed}", "--K", "25"],
  "timeout": 3600,
  "transductive": false
}
```

CLI：`python scripts/run_external.py --config YOUR_EXTERNAL_CONFIG.json --input MODEL_ONLY_CSV_DIR --output NEW_OUTPUT_DIR --seed 42`。此 CLI 的输入目录只允许上面五种 CSV，不接受真实标签或其他文件；它再次复制到临时模型目录后再调用外部脚本。

## 验证范围

本次测试只使用测试文件内现场生成的微型 NumPy 数组和 `mock subprocess`。覆盖禁止传 y、inductive图可省略test-test块、明确缺依赖和防覆盖。不下载/执行任何作者代码，不读取旧数据、旧配置 JSON、旧结果或旧指标。FINCH/CLASSIX 可选包的当前实际环境、深度方法、R/Julia模型以及真实数据数值等价性仍未验证。
