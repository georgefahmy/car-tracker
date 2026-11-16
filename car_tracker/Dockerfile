FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_PREFERENCE=system

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 \
    git \
    jq \
    libgl1 \
    libglib2.0-0t64 \
    libgomp1 \
    && if [ "$(dpkg --print-architecture)" = "amd64" ]; then \
        apt-get install -y --no-install-recommends \
            ocl-icd-libopencl1 \
            intel-opencl-icd \
            libze-intel-gpu1 \
            libze1 \
            clinfo; \
    fi \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

COPY config.yaml /tmp/addon_config.yaml
RUN git clone --depth 1 --branch main https://github.com/georgefahmy/car-tracker.git /app \
    && uv sync --frozen --no-dev

COPY run.sh /run.sh
RUN chmod a+x /run.sh

EXPOSE 8000

CMD ["/run.sh"]
