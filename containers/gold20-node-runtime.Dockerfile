FROM node:22.23.1-bookworm-slim@sha256:813a7480f28fdadac1f7f5c824bcdad435b5bc1322a5968bbbdef8d058f9dff4

# The production hidden-test evaluator applies withheld patches inside the
# sandbox, so git is part of the verifier runtime rather than a host tool.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends git=1:2.39.5-0+deb12u3 \
    && rm -rf /var/lib/apt/lists/*

RUN git --version && node --version
