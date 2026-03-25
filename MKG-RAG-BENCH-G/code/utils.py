import glob
import os

# Get first jpg from entity folder
def first_jpg_path(entity_id, base_dir):
    folder = os.path.join(base_dir, entity_id)
    files = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    return files[0] if files else None

# Load Triplet
def load_triplets(path):
    triplets = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            parts = line.split("\t")
            triplets.append(parts)
    return triplets

# Read txt file
def read_txt(file_name):
    mapping = {}
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            # Remove leading/trailing spaces/newlines
            line = line.strip()
            if not line:
                continue  # skip empty lines
            # Split into two parts: ID and description
            parts = line.split("\t")
            qid = parts[0] 
            desc = " ".join(parts[1:])
            mapping[qid] = desc
    return mapping
