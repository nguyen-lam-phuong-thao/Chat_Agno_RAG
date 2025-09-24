# scripts/create_documents.py
from pathlib import Path
import json
import pandas as pd

DATA_DIR = Path("data")
OUT_JSONL = DATA_DIR / "documents.jsonl"

def find_csv_files():
    csvs = list(DATA_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError("No CSV files found in data/ folder.")
    return csvs

def detect_files(csv_paths):
    symptom_file = None
    precaution_file = None
    for p in csv_paths:
        try:
            df = pd.read_csv(p, nrows=2)
            cols = [c.lower() for c in df.columns]
            if any(c.startswith("symptom") for c in cols):
                symptom_file = p
            if any(c.startswith("precaution") for c in cols):
                precaution_file = p
        except Exception as e:
            print(f"Warning: can't read {p.name}: {e}")
    return symptom_file, precaution_file

def read_precaution_map(precaution_path: Path):
    # return dict: disease -> [precautions]
    if not precaution_path:
        return {}
    pdf = pd.read_csv(precaution_path, dtype=str, encoding="utf-8")
    # normalize column name for disease
    if "Disease" not in pdf.columns and any("disease" in c.lower() for c in pdf.columns):
        for c in pdf.columns:
            if "disease" in c.lower():
                pdf = pdf.rename(columns={c: "Disease"})
                break
    prec_cols = [c for c in pdf.columns if c.lower().startswith("precaution")]
    prec_map = {}
    for _, r in pdf.iterrows():
        disease = str(r.get("Disease","")).strip()
        precs = []
        for c in prec_cols:
            val = r.get(c)
            if pd.notna(val):
                s = str(val).strip()
                if s and s.lower() not in ("nan","none"):
                    precs.append(s)
        prec_map[disease] = precs
    return prec_map

def build_markdown_content(disease: str, symptoms: list, precautions: list, source: str) -> str:
    """
    Build Markdown content in English.
    """
    lines = []
    lines.append(f"# Disease: {disease}")
    lines.append("")
    lines.append("**Symptoms**")
    if symptoms:
        for s in symptoms:
            lines.append(f"- {s}")
    else:
        lines.append("- No symptoms listed")
    lines.append("")
    lines.append("**Precautions**")
    if precautions:
        for p in precautions:
            lines.append(f"- {p}")
    else:
        lines.append("- No precautions listed")
    lines.append("")
    lines.append("---")
    lines.append("*Disclaimer: This information is for educational purposes only and is not a substitute for professional medical advice.*")
    lines.append(f"*Source: {source}*")
    return "\n".join(lines)

def create_documents(symptom_path: Path, precaution_map: dict):
    # read symptom csv
    sdf = pd.read_csv(symptom_path, dtype=str, encoding="utf-8")
    # normalize Disease column
    if "Disease" not in sdf.columns and any("disease" in c.lower() for c in sdf.columns):
        for c in sdf.columns:
            if "disease" in c.lower():
                sdf = sdf.rename(columns={c: "Disease"})
                break
    symptom_cols = [c for c in sdf.columns if c.lower().startswith("symptom")]
    docs = []
    for idx, r in sdf.iterrows():
        disease = str(r.get("Disease","")).strip()
        # collect symptoms
        symptoms = []
        for c in symptom_cols:
            val = r.get(c)
            if pd.notna(val):
                s = str(val).strip()
                if s and s.lower() not in ("nan","none"):
                    symptoms.append(s)
        # get precautions from map (may be empty list)
        precautions = precaution_map.get(disease, [])
        content = build_markdown_content(disease, symptoms, precautions, source=symptom_path.name)
        doc = {
            "disease": disease,
            "symptoms": symptoms,
            "precautions": precautions,
            "content": content,
            "source": symptom_path.name,
            "row_id": int(idx)
        }
        docs.append(doc)
    return docs

def save_jsonl(docs, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs:
            record = {
                "page_content": d["content"],
                "metadata": {
                    "disease": d["disease"],
                    "symptom_count": len(d["symptoms"]),
                    "precaution_count": len(d["precautions"]),
                    "source": d["source"],
                    "row_id": d["row_id"]
                }
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main():
    csvs = find_csv_files()
    print("Found CSV files:", [p.name for p in csvs])
    symptom_file, precaution_file = detect_files(csvs)
    if not symptom_file:
        raise FileNotFoundError("Can't detect the symptom CSV. Make sure the symptom file contains columns like Symptom_1,...")
    print("Symptom file detected:", symptom_file.name)
    if precaution_file:
        print("Precaution file detected:", precaution_file.name)
    else:
        print("No precaution file detected (will leave precautions empty).")

    prec_map = read_precaution_map(precaution_file)
    docs = create_documents(symptom_file, prec_map)
    save_jsonl(docs, OUT_JSONL)

    print(f"Wrote {len(docs)} documents to {OUT_JSONL.resolve()}")
    if len(docs) > 0:
        print("\n--- SAMPLE (first doc metadata + preview) ---")
        sample = docs[0]
        print("Metadata:", {"disease": sample["disease"], "symptom_count": len(sample["symptoms"]), "precaution_count": len(sample["precautions"]), "source": sample["source"], "row_id": sample["row_id"]})
        print("\nContent preview:\n")
        print(sample["content"][:800])

if __name__ == "__main__":
    main()
