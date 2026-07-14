FROM python:3.12.11-slim@sha256:47ae396f09c1303b8653019811a8498470603d7ffefc29cb07c88f1f8cb3d19f

# The production hidden-test evaluator applies withheld patches inside the
# sandbox, so git is part of the verifier runtime rather than a host tool.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git=1:2.47.3-0+deb13u1 \
    && rm -rf /var/lib/apt/lists/*

# Gold-20 verifier runtime only. Keep the complete dependency closure pinned so
# the declared test-runtime package set cannot change silently.
RUN python3 -m pip install --no-cache-dir --no-deps \
    anyio==4.13.0 \
    attrs==26.1.0 \
    certifi==2026.6.17 \
    charset-normalizer==3.4.9 \
    click==8.4.1 \
    h11==0.16.0 \
    httpcore==1.0.9 \
    httpx==0.28.1 \
    idna==3.18 \
    iniconfig==2.3.0 \
    markdown-it-py==4.2.0 \
    mdurl==0.1.2 \
    packaging==26.2 \
    pluggy==1.6.0 \
    Pygments==2.20.0 \
    pytest==9.1.0 \
    requests==2.34.2 \
    rich==15.0.0 \
    typing_extensions==4.16.0 \
    urllib3==2.7.0

RUN git --version \
    && python3 -c "import anyio, attr, attrs, certifi, charset_normalizer, click, h11, httpcore, httpx, idna, iniconfig, markdown_it, mdurl, packaging, pluggy, pygments, pytest, requests, rich, typing_extensions, urllib3"
