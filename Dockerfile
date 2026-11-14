FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir .

RUN useradd --create-home runner
USER runner
ENTRYPOINT ["medeval"]
CMD ["--help"]
