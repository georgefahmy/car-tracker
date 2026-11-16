FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_PREFERENCE=system

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python3 \
    git \
    jq \
    libgl1 \
    libglib2.0-0t64 \
    libgomp1 \
    && if [ "$(dpkg --print-architecture)" = "amd64" ]; then \
        mkdir -p /etc/apt/keyrings \
        && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&options=mr&search=0x0C0E6AF955CE463C03FC51574D098D70AFBE5E1F" \
            -o /etc/apt/keyrings/kobuk-intel-graphics.asc \
        && printf "Types: deb\nURIs: https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu/\nSuites: noble\nComponents: main\nSigned-By: /etc/apt/keyrings/kobuk-intel-graphics.asc\n" \
            > /etc/apt/sources.list.d/kobuk-intel-graphics.sources \
        && apt-get update \
        && apt-get install -y --no-install-recommends \
            ocl-icd-libopencl1 \
            intel-opencl-icd \
            libze-intel-gpu1 \
            libze1 \
            clinfo; \
    fi \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Cached dependency layer: only rebuilds when pyproject.toml or uv.lock changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev && rm -rf /root/.cache/uv

# Fast code layer: refreshes app & static assets on version bump without rebuilding .venv
COPY config.yaml /tmp/addon_config.yaml
RUN git clone --depth 1 --branch main https://github.com/georgefahmy/car-tracker.git /tmp/repo \
    && rm -rf /app/app /app/static \
    && cp -a /tmp/repo/app /tmp/repo/static /tmp/repo/yolov8n.pt /tmp/repo/pyproject.toml /tmp/repo/uv.lock /app/ \
    && uv sync --frozen --no-dev \
    && rm -rf /tmp/repo /root/.cache/uv

COPY run.sh /run.sh
RUN chmod a+x /run.sh

EXPOSE 8000

CMD ["/run.sh"]
