#!/usr/bin/env bash
# Commit every change in this project and push it to GitHub.
set -euo pipefail

cd "$(dirname "$0")"

BRANCH="main"

git add -A

if git diff --cached --quiet; then
  echo "Nothing to commit."
else
  git status --short
  echo
  msg=""
  while [ -z "$msg" ]; do
    read -r -p "Commit message: " msg
  done
  git commit -m "$msg"
fi

# Take in anything pushed from another device first, then push.
if git ls-remote --exit-code --heads origin "$BRANCH" >/dev/null 2>&1; then
  git pull --rebase --autostash origin "$BRANCH"
fi
git push -u origin "$BRANCH"
echo "Pushed to origin/$BRANCH."
