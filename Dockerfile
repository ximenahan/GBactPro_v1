# Slim CPU image for the v1 predict CLI (no conda / no Qt).
# Genome-wide reproduce still uses Option A (conda) on the host.
FROM python:3.8-slim-bookworm

WORKDIR /opt/gbactpro

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Predict-only runtime deps (matches environment.yml pins used for inference)
RUN pip install \
        "numpy==1.24.3" \
        "pandas==1.5.3" \
        "biopython==1.81" \
        "tensorflow==2.13.0" \
        "keras==2.13.1"

COPY . .

# Fail the build early if Git LFS weights were not pulled (tiny pointer files).
RUN test -f models/type1_35s10_random/saved_model.pb \
    && python - <<'PY'
from pathlib import Path
p = Path("models/type1_35s10_random/variables/variables.data-00000-of-00001")
n = p.stat().st_size
if n < 1_000_000:
    raise SystemExit(
        "Model weight file looks like a Git LFS pointer ({} bytes). "
        "Run `git lfs pull` before `docker build`.".format(n)
    )
print("OK: model weights present ({} bytes)".format(n))
PY

RUN chmod +x setup.sh test.sh scripts/*.py scripts/*.sh paper/genome_wide/run.sh \
    && mkdir -p example/output test_output results

ENTRYPOINT ["python", "/opt/gbactpro/scripts/gbactpro_predict.py"]
CMD ["-h"]
