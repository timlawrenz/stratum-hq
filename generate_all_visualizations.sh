#!/bin/bash
cd /home/tim/source/activity/stratum-hq
source .venv/bin/activate
mkdir -p examples_new

for d in example-dataset/*/; do
    base=$(basename "$d")
    # Finding the source image inside example-images that matches
    # Since the directory in example-dataset might drop the extension, let's search:
    img=$(find example-images -name "${base}.*" -type f | head -n 1)
    if [ -n "$img" ]; then
        echo "Visualizing $img with $d..."
        python scripts/visualize_example.py \
            --image "$img" \
            --stratum-dir "$d" \
            --output "examples_new/${base}_combined.png"
    fi
done
