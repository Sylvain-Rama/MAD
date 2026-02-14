########################################
# DEV STAGE (default)
########################################
FROM python:3.12-slim AS dev

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        bash \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency metadata first (for caching)
COPY pyproject.toml uv.lock* ./

# Create project virtualenv + install all deps
RUN uv sync

# Auto-activate venv when bash starts
RUN echo 'if [ -f /app/.venv/bin/activate ]; then source /app/.venv/bin/activate; fi' \
    >> /root/.bashrc

# Copy source
COPY src ./src

CMD ["bash"]


########################################
# PROD STAGE
########################################
FROM python:3.12-slim AS prod

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        bash \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock* ./

RUN uv sync --frozen --no-dev

COPY src ./src

CMD ["uv", "run", "python", "-m", "MAD"]
