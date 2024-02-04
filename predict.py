import os
import io
from pathlib import Path
import numpy as np
import tempfile
import tensorflow as tf
from PIL import Image
from typing import Any
from typing import List
from cog import BasePredictor, Input, Path
import cog

from eval import interpolator, util

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)


class Predictor(BasePredictor):
    def setup(self):
        import tensorflow as tf
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        self.interpolator = interpolator.Interpolator("pretrained_models/film_net/Style/saved_model", None)

        # Batched time.
        self.batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

    def predict(self, frame1: cog.File = Input(description="The first input frame in a file"), frame2: cog.File = Input(description="The second input frame in a file"), times_to_interpolate: int = Input(description="The second input frame", default=1)) -> Any:

        # Open files
        image_1 = Image.open(frame1)
        image_2 = Image.open(frame2)

        if not image_1.size == image_2.size:
            img1 = image_1.crop((0, 0, min(image_1.size[0], image_2.size[0]), min(image_1.size[1], image_2.size[1])))
            img2 = image_2.crop((0, 0, min(image_1.size[0], image_2.size[0]), min(image_1.size[1], image_2.size[1])))
            frame1 = 'new_frame1.jpg'
            frame2 = 'new_frame2.jpg'
            img1.save(frame1)
            img2.save(frame2)

        input_frames = [self.image_to_file_path(image_1), self.image_to_file_path(image_2)]

        # Save interpolated frames to disk
        interpolated_frames = list(
            util.interpolate_recursively_from_files(
                input_frames, times_to_interpolate, self.interpolator))

        urls = []
        output_dir = Path(tempfile.mkdtemp())
        os.makedirs(output_dir, exist_ok=True)

        for i, img in enumerate(interpolated_frames):
            out_path = os.path.join(output_dir, f"image_{i}.jpg")
            util.write_image(str(out_path), img)
            urls.append(Path(out_path))

        # Return the URLs of interpolated frames
        return urls

    def image_to_file_path(self, image):
        # Read data from the io.BytesIO object
        bytes_io = io.BytesIO()
        image.save(bytes_io, format='PNG')
        #bytes_io.seek(0)
        bytes_data = bytes_io.getvalue()

        # Create a temporary file and write the data
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(bytes_data)
            temp_file_path = temp_file.name
    
        # Convert the path of the temporary file to a string
        file_path_string = str(temp_file_path)
    
        return file_path_string