FROM continuumio/miniconda3:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN conda create -c conda-forge -y -n main python=3.12.10 \
&& conda create -c conda-forge -y -n inseq python=3.12.10 \
&& conda clean -afy

RUN git clone https://github.com/maciap/Virgil.git /src

WORKDIR /src/Virgil

RUN conda run -n main pip install --no-cache-dir -r requirements_fix.txt \
&& conda run -n inseq pip install --no-cache-dir -r xai-inseq-requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["bash", "-c", "\
conda run -n inseq python -m uvicorn inseq_service.app:app --host 0.0.0.0 --port 8001 --log-level info & \
conda run --no-capture-output -n main streamlit run Navigator.py --server.port=8501 --server.address=0.0.0.0 \
"]