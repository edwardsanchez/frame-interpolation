import os
from pathlib import Path
import numpy as np
import tempfile
import tensorflow as tf
import mediapy
from PIL import Image
import cog

from eval import interpolator, util

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)

class Predictor(cog.Predictor):
    def setup(self):
        import tensorflow as tf
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        self.interpolator = interpolator.Interpolator("pretrained_models/film_net/Style/saved_model", None)

        # Batched time.
        self.batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

        # Add the _OUTPUT_VIDEO flag
        self.output_video = False

    @cog.input(
        "frame1",
        type=Path,
        help="The first input frame",
    )
    @cog.input(
        "frame2",
        type=Path,
        help="The second input frame",
    )
    @cog.input(
        "times_to_interpolate",
        type=int,
        default=1,
        min=1,
        max=8,
        help="Controls the number of times the frame interpolator is invoked. If set to 1, the output will be the "
             "sub-frame at t=0.5; when set to > 1, the output will be the interpolation video with "
             "(2^times_to_interpolate + 1) frames, fps of 30.",
    )
    @cog.input(
        "output_video",
        type=bool,
        default=False,
        help="Set to false to get an array of image paths instead of a video."
    )
    def predict(self, frame1, frame2, times_to_interpolate, output_video):
        INPUT_EXT = ['.png', '.jpg', '.jpeg']
        assert os.path.splitext(str(frame1))[-1] in INPUT_EXT and os.path.splitext(str(frame2))[-1] in INPUT_EXT, \
            "Please provide png, jpg, or jpeg images."

        # Ensure the two images are the same size
        img1 = Image.open(str(frame1))
        img2 = Image.open(str(frame2))
        if img1.size != img2.size:
            min_width, min_height = min(img1.size[0], img2.size[0]), min(img1.size[1], img2.size[1])
            img1 = img1.crop((0, 0, min_width, min_height))
            img2 = img2.crop((0, 0, min_width, min_height))
            frame1, frame2 = 'new_frame1.png', 'new_frame2.png'
            img1.save(frame1)
            img2.save(frame2)

        input_frames = [str(frame1), str(frame2)]

        frames = list(
            util.interpolate_recursively_from_files(
                input_frames, times_to_interpolate, self.interpolator))

        if not output_video:
            # Code to handle saving frames as images and returning their paths
            image_paths = []
            temp_dir = tempfile.mkdtemp()
            for idx, frame in enumerate(frames):
                image_path = Path(temp_dir) / f'frame_{idx:03d}.png'
                util.write_image(str(image_path), frame)
                image_paths.append(str(image_path))
            return image_paths
        else:
            # Existing code to handle video generation
            print('Interpolated frames generated, saving now as output video.')
            ffmpeg_path = util.get_ffmpeg_path()
            mediapy.set_ffmpeg(ffmpeg_path)
            out_path = Path(tempfile.mkdtemp()) / "out.mp4"
            mediapy.write_video(str(out_path), frames, fps=30)
            return out_path
