"""Create a patient-level 70/15/15 split from data/manifest.csv."""
import pandas as pd
from sklearn.model_selection import train_test_split
from src.common import ROOT, load_config, set_seed, resolve_path

REQUIRED = {"image_path", "label", "patient_id", "task"}

def main():
    cfg = load_config(); set_seed(cfg["seed"])
    manifest_path = ROOT / "data" / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}. Follow README.md to create manifest.csv from the verified dataset.")
    df = pd.read_csv(manifest_path)
    missing = REQUIRED - set(df.columns)
    if missing: raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    df = df[df["task"].astype(str).str.lower().eq("spiral")].copy()
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    if not set(df.label.unique()).issubset({0,1}): raise ValueError("label must be 0=Healthy or 1=Parkinson's")
    if df[["image_path","patient_id","label"]].isna().any().any(): raise ValueError("image_path, patient_id and label must not be missing")
    df["resolved_image_path"] = df["image_path"].map(lambda x: str(resolve_path(x)))
    missing_files = [p for p in df.resolved_image_path if not __import__('pathlib').Path(p).is_file()]
    if missing_files: raise FileNotFoundError(f"{len(missing_files)} image files are missing. First: {missing_files[0]}")
    # One label per patient is required for a patient-stratified split.
    labels_per_patient = df.groupby("patient_id").label.nunique()
    if (labels_per_patient > 1).any(): raise ValueError("Some patient_id values have conflicting labels; audit the dataset labels/IDs.")
    patients = df[["patient_id","label"]].drop_duplicates()
    train_val, test = train_test_split(patients, test_size=cfg["test_fraction"], random_state=cfg["seed"], stratify=patients.label)
    val_fraction_of_train_val = cfg["validation_fraction"] / (cfg["train_fraction"] + cfg["validation_fraction"])
    train, val = train_test_split(train_val, test_size=val_fraction_of_train_val, random_state=cfg["seed"], stratify=train_val.label)
    assignment = {}
    for split_name, part in [("train",train),("validation",val),("test",test)]:
        assignment.update(dict(zip(part.patient_id, [split_name]*len(part))))
    df["split"] = df.patient_id.map(assignment)
    groups = {s:set(df.loc[df.split==s,"patient_id"]) for s in ["train","validation","test"]}
    assert not (groups['train'] & groups['validation'] or groups['train'] & groups['test'] or groups['validation'] & groups['test'])
    out = ROOT / "data" / "splits" / "split_manifest.csv"; out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("Saved:", out)
    print("Image counts:\n", df.groupby(["split","label"]).size().to_string())
    print("Patient counts:\n", patients.assign(split=patients.patient_id.map(assignment)).groupby(["split","label"]).size().to_string())
    print("Patient overlap check: PASS")

if __name__ == "__main__": main()
