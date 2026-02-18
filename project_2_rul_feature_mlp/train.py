# train.py
import os
from config import DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_N_SPLITS, DEFAULT_L2_REG
from utils import set_seed
from dataset import build_rul_dataset_with_groups
from cv_search import choose_groups_from_ablation_and_search, groupkfold_eval_subset
from features import feature_idx_from_groups
from model import train_final_model
from artifacts import RunLogger, save_rul_artifacts, save_tables_csv, save_tables_excel
from dataset_configs import DATASET_CONFIGS, DATA_ROOT

def run_dataset_pipeline(folder, dataset_name, out_dir, eol_threshold, n_splits, epochs, batch_size, verbose=0, l2_reg=DEFAULT_L2_REG):
    ds_dir = os.path.join(out_dir, dataset_name)
    os.makedirs(ds_dir, exist_ok=True)
    logger = RunLogger(os.path.join(ds_dir, "report.txt"))

    X, y, groups = build_rul_dataset_with_groups(folder, eol_threshold=eol_threshold, logger=logger)

    chosen_groups, ablation_df, search_df = choose_groups_from_ablation_and_search(
        X, y, groups,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        l2_reg=l2_reg,
        verbose=verbose,
        logger=logger
    )

    cv_summary, cv_df = groupkfold_eval_subset(
        X, y, groups, groups_to_keep=chosen_groups,
        n_splits=n_splits,
        epochs=epochs,
        batch_size=batch_size,
        l2_reg=l2_reg,
        verbose=verbose
    )

    final_epochs = int(cv_summary["median_best_epoch"])
    logger.write(f"Median best epoch from GroupKFold (chosen subset): {final_epochs}")
    logger.write()

    feat_idx = feature_idx_from_groups(chosen_groups)

    model, x_scaler, history = train_final_model(
        X, y,
        chosen_groups=chosen_groups,
        feat_idx=feat_idx,
        final_epochs=final_epochs,
        batch_size=batch_size,
        l2_reg=l2_reg,
        verbose=verbose
    )

    save_rul_artifacts(out_dir, dataset_name, model, x_scaler, feat_idx, chosen_groups, eol_threshold)

    ab_csv, se_csv = save_tables_csv(out_dir, dataset_name, ablation_df, search_df)
    xlsx_path = save_tables_excel(out_dir, dataset_name, ablation_df, search_df)

    logger.write("Saved tables:")
    logger.write(f"ablation_csv: {ab_csv}")
    logger.write(f"search_csv: {se_csv}")
    logger.write(f"tables_xlsx: {xlsx_path}")
    logger.write(f"report: {os.path.join(out_dir, dataset_name, 'report.txt')}")
    logger.write()

    return {
        "model": model,
        "history": history,
        "x_scaler": x_scaler,
        "feat_idx": feat_idx,
        "chosen_groups": chosen_groups,
        "ablation_df": ablation_df,
        "search_df": search_df,
        "cv_summary": cv_summary,
        "cv_df": cv_df,
        "final_epochs": final_epochs,
        "tables": {"ablation_csv": ab_csv, "search_csv": se_csv, "tables_xlsx": xlsx_path},
        "report": os.path.join(out_dir, dataset_name, "report.txt"),
    }

if __name__ == "__main__":
    set_seed()

    DATASET_KEY = "Lab-Li-EVE"   #dataset_name
    cfg = DATASET_CONFIGS[DATASET_KEY]

    full_path = os.path.join( DATA_ROOT, cfg["folder"])

    res = run_dataset_pipeline(
        folder=full_path,
        dataset_name=cfg["dataset_name"],
        out_dir=cfg["out_dir"],
        eol_threshold=cfg["eol_threshold"],
        n_splits=cfg["n_splits"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        verbose=cfg["verbose"],
    )


    print("Report:", res["report"])
    print("Tables:", res["tables"])
    print("Chosen groups:", res["chosen_groups"])
