import json
from pathlib import Path

import numpy as np


def build_prototypes(config, output_path):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config["model"], revision=config["revision"])
    vectors = model.encode(config["sentences"], normalize_embeddings=True, convert_to_numpy=True)
    if vectors.shape != (4, 384) or not np.isfinite(vectors).all():
        raise ValueError("Sentence encoder must return four finite 384-dimensional vectors")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, prototypes=vectors.astype(np.float32),
                        sentences=np.asarray(config["sentences"]),
                        model=np.asarray(config["model"]), revision=np.asarray(config["revision"]))
    metadata = {key: config[key] for key in ("model", "revision", "sentences")}
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return vectors


def load_prototypes(path, variant="B0", expected=None):
    with np.load(path, allow_pickle=False) as archive:
        vectors = archive["prototypes"].astype(np.float32)
        if expected is not None:
            for key in ("model", "revision"):
                if key not in archive or str(archive[key]) != expected[key]:
                    raise ValueError(f"Prototype {key} does not match configuration")
            if archive["sentences"].tolist() != expected["sentences"]:
                raise ValueError("Prototype sentence order does not match configuration")
    if vectors.shape != (4, 384) or not np.isfinite(vectors).all():
        raise ValueError("Prototype archive must contain a finite (4,384) array")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Semantic prototypes must be nonzero")
    vectors /= norms
    return np.concatenate([vectors, vectors], axis=0) if variant == "M1" else vectors
