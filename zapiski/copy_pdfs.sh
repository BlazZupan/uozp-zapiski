#!/bin/bash

# Replace with your actual remote user and host
REMOTE_PATH="file:~/anonymous/lectures/uozp/zapiski/"

# Go through immediate subdirectories
for dir in */; do
  # Check if it is indeed a directory
  [ -d "$dir" ] || continue
  
  # Find PDF files in this directory (non-recursive)
  for pdf in "$dir"*.pdf; do
    [ -f "$pdf" ] || continue
    echo "Copying $pdf to $REMOTE_PATH"
    scp "$pdf" "$REMOTE_PATH"
  done
done
