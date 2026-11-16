FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PYTHON_PREFERENCE=system

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    python3 \
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

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev && rm -rf /root/.cache/uv

COPY app ./app
COPY static ./static
COPY yolov8n.pt ./

RUN mkdir -p /app/data/snapshots

EXPOSE 8000

CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
