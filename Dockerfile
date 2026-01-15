FROM ghcr.io/astral-sh/uv:python3.12-bookworm

# Install curl and unzip for downloading benchmarks
USER root
RUN apt-get update && apt-get install -y curl unzip && rm -rf /var/lib/apt/lists/*

RUN adduser agent
USER agent
WORKDIR /home/agent

# Copy project files
COPY --chown=agent:agent pyproject.toml README.md ./

# Copy agentic-rag submodule (local dependency)
COPY --chown=agent:agent agentic-rag agentic-rag

# Copy source code
COPY --chown=agent:agent src src

# Copy benchmark data (can be overridden by download)
COPY --chown=agent:agent data data

# Copy entrypoint script
COPY --chown=agent:agent entrypoint.sh ./
USER root
RUN chmod +x /home/agent/entrypoint.sh
USER agent

# Install dependencies
RUN uv sync --no-dev

# Entrypoint downloads benchmarks then starts server
ENTRYPOINT ["/home/agent/entrypoint.sh"]
CMD ["--host", "0.0.0.0", "--port", "9009"]
EXPOSE 9009
