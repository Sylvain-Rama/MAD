########################################
# DEV STAGE
########################################
FROM python:3.12-slim AS dev

# --- Install system dependencies ---
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# --- Install uv ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# --- Set working dir & venv ---
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_LINK_MODE=copy

# --- Copy dependency metadata ---
COPY pyproject.toml uv.lock* ./

# --- Install all dev dependencies into .venv ---
RUN uv sync --dev

# --- Auto-activate venv for bash ---
RUN echo 'if [ -f $UV_HOME/bin/activate ]; then source $UV_HOME/bin/activate; fi' >> /root/.bashrc

# --- Keep container alive for interactive shell ---
CMD ["tail", "-f", "/dev/null"]



########################################
# PROD STAGE
########################################
FROM python:3.12-slim AS prod

# System deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        bash \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV UV_LINK_MODE=copy

# Copy dependency metadata
COPY pyproject.toml uv.lock* ./

# Install only runtime deps (no dev)
RUN uv sync --frozen --no-dev

# Copy source
COPY src ./src

CMD ["uv", "run", "python", "-m", "mad"]
