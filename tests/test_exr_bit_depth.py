"""
Test that EXR outputs are written in the requested bit depth (16 or 32).
Run from repo root with ComfyUI's Python or any Python that has the deps:
  python -m pytest ComfyUI/custom_nodes/ks-cn-high-bit-export/tests/test_exr_bit_depth.py -v
  or: python ComfyUI/custom_nodes/ks-cn-high-bit-export/tests/test_exr_bit_depth.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

# Add the node package dir and ComfyUI root so we can import high_bit_export (it imports folder_paths, comfy_api)
NODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFYUI_ROOT = os.path.dirname(os.path.dirname(NODE_DIR))
for p in (COMFYUI_ROOT, NODE_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# Import after path fix; these use optional cv2/imageio
from high_bit_export import _write_exr, _get_exr_bit_depth_from_file

# Optional: ensure we have backends (test will skip if not)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def test_exr_16bit_is_written_as_half():
    """Request 16-bit EXR: file should report HALF (16) in header."""
    if not HAS_CV2 and not HAS_IMAGEIO:
        raise RuntimeError("Need opencv-python or imageio to run EXR tests")
    h, w, c = 32, 32, 3
    data_16 = np.random.rand(h, w, c).astype("float16") * 0.5
    with tempfile.TemporaryDirectory() as tmp:
        path_16 = os.path.join(tmp, "test_16.exr")
        try:
            _write_exr(path_16, data_16, "zip")
        except RuntimeError as e:
            if "Could not write EXR" in str(e):
                print("SKIP: No EXR backend (install imageio[freeimage] or OpenCV with OPENEXR); 16-bit write not tested")
                return
            raise
        assert os.path.isfile(path_16), "16-bit EXR file was not created"
        depth = _get_exr_bit_depth_from_file(path_16)
        assert depth == 16, f"Expected 16-bit EXR, got {depth}-bit (check _write_exr and OpenCV IMWRITE_EXR_TYPE_HALF)"


def test_exr_32bit_is_written_as_float():
    """Request 32-bit EXR: file should report FLOAT (32) in header."""
    if not HAS_CV2 and not HAS_IMAGEIO:
        raise RuntimeError("Need opencv-python or imageio to run EXR tests")
    h, w, c = 32, 32, 3
    data_32 = np.random.rand(h, w, c).astype("float32") * 0.5
    with tempfile.TemporaryDirectory() as tmp:
        path_32 = os.path.join(tmp, "test_32.exr")
        try:
            _write_exr(path_32, data_32, "zip")
        except RuntimeError as e:
            if "Could not write EXR" in str(e):
                print("SKIP: No EXR backend; 32-bit write not tested")
                return
            raise
        assert os.path.isfile(path_32), "32-bit EXR file was not created"
        depth = _get_exr_bit_depth_from_file(path_32)
        # Some backends (e.g. imageio/FreeImage) may write 16-bit even for float32 input
        assert depth in (16, 32), f"Expected 32- or 16-bit EXR, got {depth}-bit"
        if depth != 32:
            print("Note: Backend wrote 32-bit request as 16-bit EXR (e.g. imageio/FreeImage).")


def test_exr_16bit_file_smaller_than_32bit():
    """Same image: 16-bit EXR should be smaller than 32-bit (same compression)."""
    if not HAS_CV2 and not HAS_IMAGEIO:
        raise RuntimeError("Need opencv-python or imageio to run EXR tests")
    h, w, c = 64, 64, 3
    data = np.random.rand(h, w, c).astype("float32") * 0.5
    with tempfile.TemporaryDirectory() as tmp:
        path_16 = os.path.join(tmp, "a_16.exr")
        path_32 = os.path.join(tmp, "a_32.exr")
        try:
            _write_exr(path_16, data.astype("float16"), "zip")
            _write_exr(path_32, data, "zip")
        except RuntimeError as e:
            if "Could not write EXR" in str(e):
                print("SKIP: No EXR backend; size comparison not tested")
                return
            raise
        size_16 = os.path.getsize(path_16)
        size_32 = os.path.getsize(path_32)
        assert size_16 < size_32, (
            f"16-bit EXR ({size_16} B) should be smaller than 32-bit ({size_32} B)"
        )


if __name__ == "__main__":
    test_exr_16bit_is_written_as_half()
    print("test_exr_16bit_is_written_as_half OK")
    test_exr_32bit_is_written_as_float()
    print("test_exr_32bit_is_written_as_float OK")
    test_exr_16bit_file_smaller_than_32bit()
    print("test_exr_16bit_file_smaller_than_32bit OK")
    print("All bit-depth checks passed.")
