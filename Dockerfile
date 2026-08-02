FROM continuumio/miniconda3:latest

WORKDIR /opt/gbactpro

# System deps for TensorFlow CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml .
RUN conda env create -f environment.yml \
    && conda clean -afy

# Make env default
SHELL ["conda", "run", "-n", "gbactpro", "/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/gbactpro/bin:$PATH

COPY . .

RUN chmod +x setup.sh test.sh scripts/gbactpro_predict.py paper/genome_wide/run.sh \
    && mkdir -p example/output test_output

# Default: predict CLI (override with docker run ... ./test.sh)
ENTRYPOINT ["python", "/opt/gbactpro/scripts/gbactpro_predict.py"]
CMD ["-h"]
