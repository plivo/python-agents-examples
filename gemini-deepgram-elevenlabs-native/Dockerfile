FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy from cache instead of linking (for mounted volumes)
ENV UV_LINK_MODE=copy

# Install dependencies using lockfile
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Copy application code
COPY agent.py agent.py
COPY server.py server.py
COPY system_prompt.md system_prompt.md

# Expose port
EXPOSE 8000

# Run the server
CMD ["uv", "run", "python", "server.py"]
