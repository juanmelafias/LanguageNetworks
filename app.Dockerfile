# -*- Langoo Reporting App Dockerfile -*-
#s

########################################################################################
# Stage 1: Install dependencies
########################################################################################
FROM python:3.10.0-slim-bullseye AS base

RUN mkdir -p /build/tests
RUN mkdir /src

COPY reqs /build/reqs

# Export defaults for necessary environment variables.
# ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
ENV PYTHONPATH /src

########################################################################################
# Stage 2: Copy libraries and install component dependencies
########################################################################################
FROM base AS library

RUN pip install --upgrade pip \
  && pip install -r /build/reqs/requirements.txt \
    -r /build/reqs/requirements-test.txt

RUN apt-get update \
    && apt-get install -y \
        curl \
        gcc \
        g++ \
        gnupg \
        nano \
        unixodbc-dev

# Add SQL Server ODBC Driver 17 for Debian
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
RUN curl https://packages.microsoft.com/config/debian/10/prod.list \
    > /etc/apt/sources.list.d/mssql-release.list



########################################################################################
# Stage 3: Run Application
########################################################################################
FROM library AS app

COPY app /src/app
COPY common /src/common
COPY entrypoints/app.sh /src/
COPY files /src/files

WORKDIR /src
EXPOSE 8501

RUN chmod +x app.sh
CMD ["./app.sh"]