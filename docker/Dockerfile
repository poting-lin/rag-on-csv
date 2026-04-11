# Stage 1: Build — install dependencies via Poetry
FROM python:3.13-slim AS build

RUN pip install --no-cache-dir poetry==2.1.3 poetry-plugin-export==1.9.0

WORKDIR /build

COPY pyproject.toml poetry.lock ./

RUN poetry export --without dev --format requirements.txt --output requirements.txt

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt


# Stage 2: Runtime — copy app code and venv
FROM python:3.13-slim

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY csv_qa/ csv_qa/
COPY streamlit_app.py .
COPY main.py .
COPY sample_data/ sample_data/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
