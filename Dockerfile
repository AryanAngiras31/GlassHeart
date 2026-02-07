# Dockerfile
FROM jupyter/minimal-notebook:python-3.9

# Install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Switch back to jovyan to avoid permission issues
USER ${NB_UID}

# Install Python packages
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --timeout=600 -r /tmp/requirements.txt

# Set working directory
WORKDIR /home/jovyan/work

# Expose Jupyter port
EXPOSE 8888

# Start Jupyter Notebook
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''", "--NotebookApp.allow_origin='*'", "--NotebookApp.allow_remote_access=1"]