FROM continuumio/miniconda3:latest

WORKDIR /opt/gbactpro

# System deps for TensorFlow CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml .
RUN conda env create -f environment.yml \
    && conda clean -afy

ENV PATH=/opt/conda/envs/gbactpro/bin:$PATH
ENV TF_CPP_MIN_LOG_LEVEL=2

COPY . .

# Fail the build early if Git LFS weights were not pulled (tiny pointer files).
RUN test -f models/type1_35s10_random/saved_model.pb \
    && python - <<'PY'
from pathlib import Path
p = Path("models/type1_35s10_random/variables/variables.data-00000-of-00001")
n = p.stat().st_size
if n < 1_000_000:
    raise SystemExit(
        f"Model weight file looks like a Git LFS pointer ({n} bytes). "
        "Run `git lfs pull` before `docker build`."
    )
print(f"OK: model weights present ({n} bytes)")
PY

RUN chmod +x setup.sh test.sh scripts/gbactpro_predict.py paper/genome_wide/run.sh \
        scripts/run_genome_pipeline.sh \
    && mkdir -p example/output test_output results

# Default: predict CLI (override with docker run ... ./test.sh)
ENTRYPOINT ["python", "/opt/gbactpro/scripts/gbactpro_predict.py"]
CMD ["-h"]
