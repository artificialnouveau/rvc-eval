from pythonosc import udp_client
from pythonosc import dispatcher
from pythonosc import osc_server

import os
import sys
from argparse import ArgumentParser
from logging import getLogger
from scipy.io.wavfile import read, write
from scipy.signal import resample_poly

import soundfile as sf
import numpy as np
import torch
from tqdm import tqdm
from rvc_eval.vc_infer_pipeline import VC
from rvc_eval.model import load_hubert, load_net_g

sys.path.append(os.path.join(os.path.dirname(__file__), "../rvc/"))

logger = getLogger(__name__)

def print_handler(address, *args):
    print(f"Received message from {address}: {args}")

def receive_osc(unused_addr, model, input_file_path, output_file_path):
    global args
    args.model = model
    args.input_file = input_file_path
    args.output_file = output_file_path
    main(args)

dispatcher = dispatcher.Dispatcher()
dispatcher.map("/OSC_2py", print_handler)

server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 1111), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()

def resample_audio(audio, original_sr, target_sr):
    from math import gcd

    # Calculate up and down sampling ratios
    g = gcd(original_sr, target_sr)
    up = target_sr // g
    down = original_sr // g

    return resample_poly(audio, up, down)


def main(args):
    is_half = not args.float and args.device != "cpu"
    device = torch.device(args.device)
    print('Device used: ', device)

    hubert_model = load_hubert(args.hubert, is_half, device)
    net_g, sampling_ratio = load_net_g(args.model, is_half, device)

    repeat = 3 if is_half else 1
    repeat *= args.quality  # 0 or 3
    sid = 0
    f0_up_key = args.f0_up_key
    f0_method = args.f0_method
    vc = VC(sampling_ratio, device, is_half, repeat)

    audio_output_chunks = []

    with sf.SoundFile(args.input_file, 'r') as f:
        input_frame_rate = f.samplerate
        frames_per_buffer = input_frame_rate * args.buffer_size // 1000
        
        with tqdm(total=len(f), desc="Processing", unit="sample") as pbar:
            while True:
                audio_input = f.read(frames=frames_per_buffer)

                if len(audio_input) == 0:  # End of file
                    break

                if len(audio_input.shape) > 1 and audio_input.shape[1] > 1:
                    audio_input = np.mean(audio_input, axis=1)

                audio_input = audio_input.astype(np.float64)

                audio_output = (
                    vc.pipeline(
                        hubert_model,
                        net_g,
                        sid,
                        audio_input,
                        f0_up_key,
                        f0_method,
                    )
                )

                audio_output_chunks.append(audio_output)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if np.isnan(audio_output).any():
                    continue

                pbar.update(len(audio_input))
    
    audio_output_full = np.concatenate(audio_output_chunks)

    # Save the output
    sf.write(args.output_file, audio_output_full.astype('float32'), 44100)  # Save at 44.1kHz rate
        
    # Specify the IP address of the remote host
    remote_ip = "192.168.2.110"  # Replace with the IP address of the remote host

    # Create a client to send OSC messages, targeting the remote host on port 5005
    sender = udp_client.SimpleUDPClient(remote_ip, 6666)
    # sender = udp_client.SimpleUDPClient("127.0.0.1", 6666)
    sender.send_message("/OSC_2max", args.output_file)

    # Send OSC command to Max to notify that file creation is complete
    # client = udp_client.SimpleUDPClient('127.0.0.1', 6448)  # Assuming Max is listening on port 6448 at localhost
    # client.send_message("/fileComplete", args.output_file)  # Sending the path as a message for confirmation


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
parser.add_argument("--buffer-size", type=int, default=1000, help="buffering size in ms")


if __name__ == "__main__":
    args = parser.parse_args()
    logger.setLevel(args.log_level)
    main(args)

