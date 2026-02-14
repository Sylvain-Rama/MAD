########################################
# DEV STAGE (default)
########################################
FROM python:3.12-slim AS dev

# --- system deps ---
RUN apt-get update \
    && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# --- install uv ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy dependency metadata first (build cache)
COPY pyproject.toml uv.lock* ./

# Install ALL deps including dev tools
RUN uv sync

# Copy source
COPY src ./src

# Interactive dev shell by default
CMD ["bash"]


########################################
# PROD STAGE
########################################
FROM python:3.12-slim AS prod

RUN apt-get update \
    && apt-get install -y curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

COPY pyproject.toml uv.lock* ./

# Install ONLY runtime deps
RUN uv sync --frozen --no-dev

COPY src ./src

CMD ["uv", "run", "python", "-m", "yourpkg"]
