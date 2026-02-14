########################################
# DEV STAGE (default)
########################################
FROM python:3.12-slim AS dev

# --- Install system deps ---
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        bash \
    && rm -rf /var/lib/apt/lists/*

# --- Install uv ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# --- Set UV_HOME to a container-only path (avoids Windows .venv issues) ---
ENV UV_HOME=/root/.uv
ENV PATH="$UV_HOME/bin:$PATH"

WORKDIR /app

# Copy dependency metadata first (for caching)
COPY pyproject.toml uv.lock* ./

# Install runtime + dev deps
RUN uv sync

# Auto-activate uv venv when bash starts
RUN echo 'if [ -f $UV_HOME/bin/activate ]; then source $UV_HOME/bin/activate; fi' >> /root/.bashrc

# Copy project source
COPY src ./src

# Default: interactive bash
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

# Copy dependency metadata
COPY pyproject.toml uv.lock* ./

# Install only runtime deps (no dev)
RUN uv sync --frozen --no-dev

# Copy source
COPY src ./src

CMD ["uv", "run", "python", "-m", "mad"]
