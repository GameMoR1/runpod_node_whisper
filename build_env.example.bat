@echo off
REM Copy to build_env.bat and fill values.

REM Where to push:
REM   Docker Hub: leave REGISTRY empty, set NAMESPACE to your dockerhub username/org
REM   GHCR: set REGISTRY=ghcr.io, set NAMESPACE to your github user/org
set "REGISTRY="
set "NAMESPACE=YOUR_NAMESPACE"

REM Repo name in the registry
set "IMAGE_NAME=runpod-node-whisper"

REM Default tag
set "TAG=latest"

REM Auto-push when running build_container.bat without args
set "PUSH=1"

REM Optional: auto-login (or just run `docker login` once)
REM set "DOCKER_USERNAME=YOUR_USERNAME"
REM set "DOCKER_PASSWORD=YOUR_TOKEN_OR_PASSWORD"

