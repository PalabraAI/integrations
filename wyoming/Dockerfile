FROM python:3.12-slim AS build
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src src
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN useradd --system --uid 10001 --create-home wyoming
COPY --from=build /install /usr/local
USER wyoming
EXPOSE 10300
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import socket; socket.create_connection(('127.0.0.1', 10300), timeout=2).close()" || exit 1
ENTRYPOINT ["wyoming-palabra"]
CMD ["--uri", "tcp://0.0.0.0:10300"]
