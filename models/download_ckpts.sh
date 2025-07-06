#!/bin/bash

cat << "EOF"
===============================
   |\---/|
   | ,_, |
    \_`_/-..----.
 ___/ `   ' ,""+ \  sk
(__...'   __\    |`.___.';
  (_,...'(_,.`__)/'.....+
EOF

echo "Downloading checkpoints for InkLayer! Please wait..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# For InkLayer GroundingDINO
# TODO: Add the correct link for InkLayer GroundingDINO

# For segment-anything
wget -O "$SCRIPT_DIR/sam_vit_h_4b8939.pth" \
"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# For depth-anything
wget -O "$SCRIPT_DIR/depth_anything_v2_vitb.pth" \
"https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"

# For InkLayer GroundingDINO
wget -O "$SCRIPT_DIR/inklayer_gdino.pth" \
"https://huggingface.co/miatang13/InkLayer/resolve/main/inklayer_gdino.pth"

echo "Downloaded all checkpoints! (^_^)"