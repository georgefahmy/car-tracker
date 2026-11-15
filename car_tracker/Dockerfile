FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    jq \
    libgl1 \
    libglib2.0-0 \
    && if [ "$(dpkg --print-architecture)" = "amd64" ]; then apt-get install -y --no-install-recommends ocl-icd-libopencl1 intel-opencl-icd || true; fi \
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
