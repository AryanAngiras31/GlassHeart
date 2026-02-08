FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# You still COPY . . here so the image is self-contained for production deployment
COPY . . 

EXPOSE 8501

# The address=0.0.0.0 is required for Docker networking
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]