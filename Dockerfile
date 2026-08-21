# See README.md "Deployment -> Option B" for build/run instructions and
# the persistent-volume note (data/finmate.db + data/chroma_store are
# plain files on disk -- mount a volume at /app/data or they won't
# survive a container restart/redeploy).
FROM python:3.12-slim
WORKDIR /app

# curl is needed for the HEALTHCHECK below -- not present in the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
