#!/bin/bash

OUTPUT_ZIP="project_package.zip" 

echo "Zipping current Git repository into $OUTPUT_ZIP..."

zip -r "$OUTPUT_ZIP" . \
  -x ".git/*" \
  -x "*.DS_Store" \
  -x "*.pyc" \
  -x "__pycache__/*" \
  -x "$OUTPUT_ZIP"

echo "Git repository packaged successfully into $OUTPUT_ZIP."
