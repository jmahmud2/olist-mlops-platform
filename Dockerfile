FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies including uvicorn
RUN pip install --no-cache-dir -r requirements.txt uvicorn

# Copy the rest of the application
COPY . .

# Make sure uvicorn is in PATH
RUN pip show uvicorn

EXPOSE 8000

CMD ["uvicorn", "api.serve:app", "--host", "0.0.0.0", "--port", "8000"]