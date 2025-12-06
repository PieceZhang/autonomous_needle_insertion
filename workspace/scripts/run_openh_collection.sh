#!/usr/bin/env bash
set -e

echo "Launching rqt with a perspective file for OpenH dataset collection ..."
echo

# Run rqt with the specified perspective file
rqt --perspective-file ./rqt_perspective_file/openh.perspective \
  > ./log/rqt_dataset_collection.log 2>&1 &