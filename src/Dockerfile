FROM python:3.12-slim

WORKDIR /app

# Install deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code (main.py launches tools.py as a child MCP server over stdio).
COPY tools.py skills.py agents.py main.py ./

CMD ["python", "main.py"]
