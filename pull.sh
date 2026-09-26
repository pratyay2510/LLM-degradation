#!/usr/bin/env bash
# Pull the latest changes from GitHub into this project.
set -euo pipefail

cd "$(dirname "$0")"

REMOTE_URL="https://github.com/pratyay2510/LLM-degradation.git"
BRANCH="main"

# First run on a new device: turn this folder into a clone of the repo.
if [ ! -d .git ]; then
  git init -b "$BRANCH"
  git remote add origin "$REMOTE_URL"
  git fetch origin "$BRANCH"
  git reset --mixed "origin/$BRANCH"
  git branch --set-upstream-to="origin/$BRANCH" "$BRANCH"
  echo "Linked this folder to $REMOTE_URL. Local files that differ from GitHub are left as uncommitted changes."
  exit 0
fi

git pull --rebase --autostash origin "$BRANCH"
echo "Up to date with origin/$BRANCH."
