# Source provenance

Only source files were inspected while preparing this package. None of their
data paths, cached predictions, figures, metrics, API keys or runtime state was
opened or used. The names below identify development sources; the package does
not import or depend on those directories.

| Packaged component | Source reviewed |
| --- | --- |
| Strict Gaussian/SMLE updates and initializers | `realdata_current_20260926/method_code/strict_smle.py` |
| Log-domain sufficient statistics | `XA_realdata_comparison_20261004/source_snapshots/gm3l/stable_em.py` |
| FINAL and rectangular star prediction | `XA_realdata_comparison_20261004/adapters/gm3l.py` |
| Full-graph protocol and native starts/alignment | `realdata_llm_full_20260929/run_full_llm_a.py` |
| Full-covariance GMM | `realdata_current_20260926/run_x_only_baseline.py` |
| Tied GMM | `realdata_gmm_tied_fixed50_20261012/fit_tied.py` |
| Training-label oracle and parameter deviations | `realdata_current_20260926/method_code/complete_data.py`, `run_ten_class_pilot.py` |
| ID-based cleaning/input preparation | `outputs/realdata_clean_20260922/prepare_clean_data.py` |
| Local embedding recipe | `BGE_embedding_setup_20260922/embed_hotline.py`, `load_embeddings.py` |
| Binary must-link parsing/completeness | `mustlink_A_setup_20260922/mustlink_a.py` |
| Train/test graph block layout | `XA_realdata_comparison_20261008_figure3/prepare_data.py` |
| Classical comparisons and continuous scores | `Xonly_realdata_comparison_20261009/adapters/classical.py`, `auc_adapters/classical_scores.py` |
| Metric conventions | `XA_realdata_comparison_20261004/evaluate.py`, `XA_realdata_comparison_20261008_figure3/evaluate.py`, `Xonly_realdata_comparison_20261009/auc_supplement.py` |
| Comparison figure organization | `XA_realdata_comparison_20261008_figure3/figure3_fixed50_20261009/plot_fixed50.py`, `realdata_auc_xa_six_methods_20261014/plot_auc.py` |

This list does not assert a license for upstream software. Native scientific
implementations are separated from optional third-party packages; retain the
latter's original attribution and license when installing or redistributing.
Detailed comparison-method provenance and implementation distinctions are in
`BASELINES.md` and the adapter-specific documents.
