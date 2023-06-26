import numpy as np
from pympler.asizeof import asizeof
import json
import io
import base64
from PIL import Image

# Receive and decode bytes to numpy array
def receive_decode_bytes_to_numpy_array(j_dumps:str) -> np.array:
    # Convert Base 64 representation to byte representation
    compressed_data = base64.b64decode(j_dumps)
    # Read byte array to an Image
    im = Image.open(io.BytesIO(compressed_data))
    # Return Image to numpy array format
    return np.array(im)
