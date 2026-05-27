@echo off
setlocal enabledelayedexpansion

REM Optional config file (same folder):
REM   build_env.bat can set:
REM     REGISTRY=docker.io  (or ghcr.io, etc)
REM     NAMESPACE=myuser    (docker hub user or org; for ghcr.io: owner)
REM     IMAGE_NAME=runpod-node-whisper
REM     TAG=latest
REM     PUSH=1
REM     DOCKER_USERNAME=...
REM     DOCKER_PASSWORD=...  (or set via environment)
set "SCRIPT_DIR=%~dp0"
if exist "%SCRIPT_DIR%build_env.bat" (
  call "%SCRIPT_DIR%build_env.bat"
)

REM Usage:
REM   build_container.bat [image] [tag] [push]
REM Examples:
REM   build_container.bat myrepo/runpod-node-whisper latest push
REM   build_container.bat  (uses build_env.bat / env vars)

set "ARG_IMAGE=%~1"
set "ARG_TAG=%~2"
set "ARG_PUSH=%~3"

if "%ARG_IMAGE%"=="" (
  if "%IMAGE%"=="" (
    if "%IMAGE_NAME%"=="" set "IMAGE_NAME=runpod-node-whisper"
    if "%NAMESPACE%"=="" (
      echo ERROR: NAMESPACE is not set.
      echo - Set it in build_env.bat or as env var, for example: set NAMESPACE=mydockerhubuser
      exit /b 1
    )
    if "%REGISTRY%"=="" (
      REM Default to Docker Hub namespace/repo (docker.io is implicit)
      set "IMAGE=%NAMESPACE%/%IMAGE_NAME%"
    ) else (
      set "IMAGE=%REGISTRY%/%NAMESPACE%/%IMAGE_NAME%"
    )
  )
) else (
  set "IMAGE=%ARG_IMAGE%"
)

set "TAG=%ARG_TAG%"
if "%TAG%"=="" (
  if not "%TAG%"=="" (
    REM no-op (kept for readability)
  )
)
if "%TAG%"=="" set "TAG=latest"

set "DO_PUSH=%ARG_PUSH%"
if "%DO_PUSH%"=="" (
  if "%PUSH%"=="1" set "DO_PUSH=push"
)

echo.
echo Target image: %IMAGE%:%TAG%
echo.

if /I "%DO_PUSH%"=="push" (
  if not "%DOCKER_USERNAME%"=="" (
    if not "%DOCKER_PASSWORD%"=="" (
      echo Logging in as %DOCKER_USERNAME% ...
      echo %DOCKER_PASSWORD%| docker login -u "%DOCKER_USERNAME%" --password-stdin
      if errorlevel 1 (
        echo ERROR: docker login failed.
        exit /b 1
      )
    )
  )
)

echo.
echo Building Docker image: %IMAGE%:%TAG%
echo Dockerfile: %CD%\Dockerfile
echo.

docker build -t "%IMAGE%:%TAG%" .
if errorlevel 1 (
  echo ERROR: docker build failed.
  exit /b 1
)

echo.
echo OK: built %IMAGE%:%TAG%

if /I "%DO_PUSH%"=="push" (
  echo.
  echo Pushing %IMAGE%:%TAG% ...
  docker push "%IMAGE%:%TAG%"
  if errorlevel 1 (
    echo ERROR: docker push failed.
    exit /b 1
  )
  echo OK: pushed %IMAGE%:%TAG%
)

echo.
set "RUNPOD_IMAGE=%IMAGE%:%TAG%"
REM If user omitted registry for Docker Hub, RunPod usually wants explicit docker.io prefix.
echo %IMAGE% | findstr /I /B "docker.io/" >nul
if errorlevel 1 (
  echo %IMAGE% | findstr /I "^[^/][^/]*/[^/][^/]*" >nul
  if not errorlevel 1 (
    set "RUNPOD_IMAGE=docker.io/%IMAGE%:%TAG%"
  )
)
echo RunPod imageName: !RUNPOD_IMAGE!
echo RunPod (copy/paste): !RUNPOD_IMAGE!
echo Done.
endlocal

