FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Copy the lockfile and pyproject.toml
COPY uv.lock pyproject.toml /app/

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# Place the application in the virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application source code
COPY src /app/src
COPY pyproject.toml /app/

# Install the project itself (editable mode is default for uv sync without --no-install-project, 
# but we used --no-install-project above to cache deps layer, so now we install the app)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

EXPOSE 8000

# Disable Python output buffering for real-time logs
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
