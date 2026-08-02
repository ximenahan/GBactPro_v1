# Models

TensorFlow / Keras **SavedModel** directories. Each folder contains:

- `saved_model.pb`, `variables/`, `keras_metadata.pb` — loadable with `tf.keras.models.load_model`
- `metadata.json` — maxlen, padding, length range, provenance

## Git LFS

Weight shards (`variables/variables.data-00000-of-00001`) are ~127 MB each and exceed GitHub’s 100 MB blob limit. This repo tracks them with **Git LFS** (see `.gitattributes`).

```bash
git lfs install
git clone <repo-url>
cd GBactpro
git lfs pull
```

If LFS objects are missing, `./setup.sh` will report incomplete models.

## Which model to use

| Folder | Public API | Maxlen | Padding |
|--------|------------|--------|---------|
| `type1_35s10_random` | `scripts/gbactpro_predict.py` | 31 | post |
| `type1_pre_max29bp_random` | `paper/genome_wide/run.sh` | 29 | pre (training) |
