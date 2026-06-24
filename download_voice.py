import urllib.request
import os

print("Downloading ultra-realistic voice model weights (approx. 300MB)...")
# Fixed URL pointing to the active model-files release tag
model_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
urllib.request.urlretrieve(model_url, "kokoro-v0.19.onnx")

print("Downloading voice geometry profile...")
# Fixed URL pointing to the new bin mapping file
voices_url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"
urllib.request.urlretrieve(voices_url, "voices.json")

print("Done! Assets saved successfully.")