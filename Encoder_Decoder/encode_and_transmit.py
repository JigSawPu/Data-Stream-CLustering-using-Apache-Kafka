import numpy as np
import json
import io
import base64
from PIL import Image

# Lambda function to represent memory in Mb
# Encode and transmit Numpy Array in bytes
def encode_and_transmit_numpy_array_in_bytes(numpy_array:np.array) -> str:
    # Create a Byte Stream Pointer
    compressed_file = io.BytesIO()
    # Use PIL JPEG reduction to save the image to bytes
    Image.fromarray(numpy_array).save(compressed_file, format="JPEG")
    # Set index to start position
    compressed_file.seek(0)
    # Convert the byte representation to base 64 representation for REST Post
    return json.dumps(base64.b64encode(compressed_file.read()).decode())

