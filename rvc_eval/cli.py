import os
import sys
from argparse import ArgumentParser
from logging import getLogger
from scipy.io.wavfile import read, write'

import soundfile as sf
import numpy as np
import torch
from tqdm import tqdm
from rvc_eval.vc_infer_pipeline import VC
from rvc_eval.model import load_hubert, load_net_g

sys.path.append(os.path.join(os.path.dirname(__file__), "../rvc/"))

logger = getLogger(__name__)

def main(args):
    is_half = not args.float and args.device != "cpu"
    device = torch.device(args.device)

    hubert_model = load_hubert(args.hubert, is_half, device)
    net_g, sampling_ratio = load_net_g(args.model, is_half, device)

    repeat = 3 if is_half else 1
    repeat *= args.quality  # 0 or 3
    sid = 0
    f0_up_key = args.f0_up_key
    f0_method = args.f0_method
    vc = VC(sampling_ratio, device, is_half, repeat)

    # Read input audio file parameters
    input_frame_rate = 16000
    frames_per_buffer = input_frame_rate * args.buffer_size // 1000

    # Placeholder to store the processed audio
    audio_output_full = np.empty((0,))

    total_iterations = len(sf.SoundFile(args.input_file)) // frames_per_buffer
    
    # Read the input audio file in chunks with a progress bar
    with sf.SoundFile(args.input_file, 'r') as f, tqdm(total=total_iterations, desc="Processing", unit="chunk") as pbar:
        while True:
            audio_input = f.read(frames=frames_per_buffer)
            
            if len(audio_input) == 0:  # End of file
                break

            # Check for multi-channel audio and convert to mono
            if len(audio_input.shape) > 1 and audio_input.shape[1] > 1:
                audio_input = np.mean(audio_input, axis=1)
            
            # Convert to float32
            audio_input = audio_input.astype(np.float32)

            audio_output = (
                vc.pipeline(
                    hubert_model,
                    net_g,
                    sid,
                    audio_input,  # don't pad here, leave it to the pipeline function
                    f0_up_key,
                    f0_method,
                )
                .cpu()
                .float()
                .numpy()
            )

            # Trim the output to match the input's length
            if len(audio_output) > len(audio_input):
                audio_output = audio_output[:len(audio_input)]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if np.isnan(audio_output).any():
                continue

            audio_output_full = np.concatenate((audio_output_full, audio_output))
            
            # Increment the progress bar
            pbar.update(1)

    # Write the processed audio to the output file using the original sample rate
    sf.write(args.output_file, audio_output_full, input_frame_rate)


parser = ArgumentParser()
parser.add_argument("-l", "--log-level", type=str, default="WARNING")
parser.add_argument(
    "-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)
parser.add_argument("-m", "--model", type=str, required=True)
parser.add_argument("--hubert", type=str, default="models/hubert_base.pt")
parser.add_argument("--float", action="store_true")
parser.add_argument("-q", "--quality", type=int, default=1)
parser.add_argument("-k", "--f0-up-key", type=int, default=0)
parser.add_argument("--f0-method", type=str, default="pm", choices=("pm", "harvest"))
parser.add_argument("--input-file", type=str, required=True, help="Path to input audio file")
parser.add_argument("--output-file", type=str, required=True, help="Path to save processed audio file")
parser.add_argument("--buffer-size", type=int, default=10000, help="buffering size in ms")


if __name__ == "__main__":
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    main(args)

