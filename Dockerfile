# ============================================================
# Dockerfile — Project Sentinel
# ============================================================
# A Dockerfile is a recipe that tells Docker how to build a
# self-contained "container image" for our application.
#
# WHAT IS A CONTAINER?
#   A container is like a tiny, isolated computer inside your
#   computer. It has its own filesystem, Python installation,
#   and running processes — completely separate from your machine.
#   This means "it works on my machine" IS "it works everywhere".
#
# HOW TO BUILD THIS IMAGE:
#   docker build -t project-sentinel .
#   (The dot means "use the current directory as the build context")
#
# HOW TO RUN THE CONTAINER:
#   docker run -p 7860:7860 project-sentinel
#   (Maps port 7860 on your machine to port 7860 inside the container)
#
# WHY PORT 7860?
#   Hugging Face Spaces REQUIRES port 7860. This is mandatory.
#   If you use any other port, the HF Space will not work.
#
# BASE IMAGE CHOICE:
#   python:3.11-slim uses Python 3.11 (NOT 3.10 — the container
#   environment uses 3.11). "slim" means it's a minimal image
#   without extra tools — smaller, faster, more secure.
# ============================================================


# ── STAGE 1: Choose the base image ──────────────────────────
# Every Dockerfile starts with FROM — it picks the starting point.
# We're starting from an official Python 3.11 image on Debian Linux.
# "slim" = stripped down, no build tools, no docs — saves ~200MB.
FROM python:3.11-slim


# ── STAGE 2: Set the working directory ──────────────────────
# All subsequent commands (COPY, RUN, CMD) will happen inside /app.
# This is the folder our app lives in inside the container.
# If /app doesn't exist, Docker creates it automatically.
WORKDIR /app


# ── STAGE 3: Copy requirements FIRST (for layer caching) ────
# Docker builds images in "layers" — one layer per instruction.
# If a layer hasn't changed, Docker reuses the cached version.
#
# WHY copy requirements.txt before copying the rest of the code?
# Because pip install is SLOW (~30-60 seconds). If we copy
# everything first, then ANY code change triggers a full
# pip install again. By copying requirements.txt first:
#   - If requirements.txt didn't change → cached pip install layer reused ✓
#   - If only app.py changed → only the COPY . step re-runs ✓
# This makes rebuilds much faster during development.
COPY requirements.txt .


# ── STAGE 4: Install Python dependencies ────────────────────
# Run pip install inside the container.
# --no-cache-dir: don't store the downloaded packages in a cache.
#   This reduces image size — we only need the installed packages,
#   not the download cache that pip normally keeps.
# -r requirements.txt: install everything listed in that file.
RUN pip install --no-cache-dir -r requirements.txt


# ── STAGE 5: Copy the rest of the application code ──────────
# Now that dependencies are installed, copy all project files.
# The first dot "." means "everything in the current directory on your machine".
# The second dot "." means "into the WORKDIR (/app) inside the container".
# This copies: app.py, env.py, models.py, inference.py, openenv.yaml, etc.
COPY . .


# ── STAGE 6: Expose the port ────────────────────────────────
# EXPOSE tells Docker (and humans reading this file) that the
# container listens on port 7860. This is documentation + a signal
# to Docker networking — it does NOT automatically publish the port
# to your machine (that's what -p 7860:7860 does in docker run).
# Port 7860 is MANDATORY for Hugging Face Spaces.
EXPOSE 7860


# ── STAGE 7: Define the startup command ─────────────────────
# CMD is what runs when someone starts the container.
# This launches our FastAPI app using uvicorn (the ASGI server).
#
# Breaking down the command:
#   uvicorn       — the server that runs FastAPI apps
#   app:app       — "in the file app.py, find the variable called app"
#   --host 0.0.0.0 — listen on ALL network interfaces, not just localhost.
#                    This is required inside Docker — without it, the server
#                    is only reachable from inside the container itself.
#   --port 7860   — listen on port 7860 (mandatory for HF Spaces)
#
# CMD uses JSON array format ["cmd", "arg1", "arg2"] — this is the
# recommended form because it avoids shell interpretation issues.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
