FROM ghcr.io/astral-sh/uv:python3.12-bookworm

RUN adduser agent
USER agent
WORKDIR /home/agent

# Copy project files
COPY --chown=agent:agent pyproject.toml README.md ./

# Copy agentic-rag submodule (local dependency)
COPY --chown=agent:agent agentic-rag agentic-rag

# Copy source code
COPY --chown=agent:agent src src

# Copy benchmark data
COPY --chown=agent:agent data data

# Install dependencies
RUN uv sync --no-dev

# Server entrypoint
ENTRYPOINT ["uv", "run", "python", "-m", "src.server"]
CMD ["--host", "0.0.0.0", "--port", "9009"]
EXPOSE 9009
