from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
import traceback

import time
import os
import sys
from argparse import ArgumentParser
from logging import getLogger
from scipy.io.wavfile import read, write
from scipy.signal import resample_poly
from threading import Thread

from threading import Thread

import soundfile as sf
import numpy as np
import torch
from tqdm import tqdm
import traceback

import warnings
warnings.filterwarnings("ignore")

from rvc_eval.vc_infer_pipeline import VC
# from rvc_eval.vc2_infer_pipeline import VC2
from rvc_eval.model import load_hubert, load_net_g
import json
import whisper


sys.path.append(os.path.dirname(__file__))
from speech_analysis import analyze_audio

# sys.path.append(os.path.join(os.path.dirname(__file__), "../rvc/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..\\rvc\\"))

logger = getLogger(__name__)

def print_handler(address, *args):
    print(f"Received message from {address}: {args}")

osc_args = {
    "models": [],
    "input_files": [],
    "output_files": []
}

def set_all_paths(address, args_string, analyze=True):  # 'analyze' parameter
    global osc_args
    
    if args_string.startswith("'") and args_string.endswith("'"):
        args_string = args_string[1:-1]
    if 'Macintosh HD:' in args_string:
        args_string = args_string.replace('Macintosh HD:', '')
        
    paths = args_string.split(", ")
    input_files = []
    output_files = []
    models = []

    try:
        # The first path is always input
        input_file = paths[0]
        
        # For the remaining paths, order is: model1, output1, model2, output2, ...
        for i in range(1, len(paths)-1, 2):
            models.append(paths[i])
            output_files.append(paths[i + 1])

        # Ensure the input_files list has the same length as models and output_files
        input_files = [input_file] * len(models)
        osc_args["input_files"] = input_files
        osc_args["output_files"] = output_files
        osc_args["models"] = models

        print("input_files: ", osc_args["input_files"])
        print("output_files: ", osc_args["output_files"])
        print("models: ", osc_args["models"])


    except IndexError:
        print("Incorrect sequence of arguments received. Expecting input_path, followed by alternating model_path and output_path.")

    try:
        if analyze:
            analyze_audio(input_file)
    except IndexError:
        print("Incorrect sequence of arguments received. Expecting input_path, followed by alternating model_path and output_path.")


#     while True:
#         server.handle_request()

#         # Run the main function for each model, input, and output path
#         for model_path, input_path, output_path in zip(osc_args["models"], osc_args["input_files"], osc_args["output_files"]):
#             args.model = model_path.replace('"', '')
#             args.input_file = input_path.replace('"', '')
#             args.output_file = output_path.replace('"', '')
#             main(args)
        
#         # Optionally add a short sleep to not overload the CPU (or you can remove it if not needed)
#         time.sleep(1)
def run_osc_server(args):
    disp = Dispatcher()
    disp.map("/max2py", set_all_paths)  # One OSC address to set all paths

    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 1111), disp)
    print(f"Serving on {server.server_address}")

    def handle_requests():
        while True:
            server.handle_request()
            
            # Run the main function for each model, input, and output path
            for model_path, input_path, output_path in zip(osc_args["models"], osc_args["input_files"], osc_args["output_files"]):
                args.model = model_path.replace('"', '')
                args.input_file = input_path.replace('"', '')
                args.output_file = output_path.replace('"', '')
                main(args)

    # Run the server in a separate thread
    thread = Thread(target=handle_requests)
    thread.start()
    

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
    #vc = VC2(sampling_ratio, device, is_half, repeat)

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
    

        # Send OSC command only if --use-osc argument is provided
        if args.use_osc:
            print('sending done command to "127.0.0.1", 6666')
            # Create a client to send OSC messages, targeting the remote host on port 5005
            sender = udp_client.SimpleUDPClient("127.0.0.1", 6666) #Remote: 192.168.2.110
            _out = args.output_file
            _mod = args.model
            message = 'output file: '+_out+' with '+_mod+' is done.'
            sender.send_message("/py2max/gen_done", message)
        
    except Exception as e:
        # If an error occurs, print the error and send an OSC message
        error_message = str(e) + '\n' + traceback.format_exc()
        print("Error:", error_message)


    # Save the output
    sf.write(args.output_file, audio_output_full.astype('float32'), 44100)  # Save at 44.1kHz rate
    print('Done with processing and I have now saved the file here: ', args.output_file)

    # Send OSC command only if --use-osc argument is provided
    if args.use_osc:
        # Create a client to send OSC messages, targeting the remote host on port 5005
        sender = udp_client.SimpleUDPClient("127.0.0.1", 6666) #Remote: 192.168.2.110
        _out = args.output_file
        _mod = args.model
        message = 'output file: '+_out+' with '+_mod+' is done.'
        sender.send_message("/py2max/gen_done", message)

parser = ArgumentParser()
parser.add_argument("--use-osc", action="store_true", help="Run in OSC mode.")
parser.add_argument("-m", "--model", type=str, required=False, help="Path to model file")
parser.add_argument("--input-file", type=str, required=False, help="Path to input audio file")
parser.add_argument("--output-file", type=str, required=False, help="Path to save processed audio file")
parser.add_argument("-l", "--log-level", type=str, default="WARNING")
parser.add_argument(
    "-d", "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
)
parser.add_argument("--hubert", type=str, default="models/hubert_base.pt")
parser.add_argument("--float", action="store_true")
parser.add_argument("-q", "--quality", type=int, default=1)
parser.add_argument("-k", "--f0-up-key", type=int, default=0)
parser.add_argument("--f0-method", type=str, default="pm", choices=("pm", "harvest"))
parser.add_argument("--buffer-size", type=int, default=1000, help="buffering size in ms")
parser.add_argument("--analyze", action="store_true", help="Analyze the input audio file.")


if __name__ == "__main__":
    args = parser.parse_args()
    logger.setLevel(args.log_level)

    if args.analyze:
        if not args.input_file and not args.use_osc:
            print("The --input-file option is required for analysis when not using OSC mode.")
            sys.exit(1)
        if args.input_file:
            json_result = analyze_audio(args.input_file)
            print(f"Analysis saved in: {json_result}")

    if args.use_osc:
        run_osc_server(args)
    else:
        if not args.model or not args.input_file or not args.output_file:
            print("When not using OSC mode, -m/--model, --input-file, and --output-file are required.")
            sys.exit(1)
        main(args)
