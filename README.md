# RVC Eval Simplified
## Origins

This code was forked from [esnya/rvc-eval](https://github.com/esnya/rvc-eval)

## Description

This project is a simplified implementation of the evaluation part of the [Retrieval-based Voice Conversion (RVC)](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI) system. It aims to make it easy to use in any Python code by removing unnecessary components. As a sample, a command-line interface (CLI) is provided for real-time voice conversion.

## Features

- Simplified RVC evaluation
- Easy integration into any Python code
- Real-time voice conversion via CLI

## Project Name

RVC Eval Simplified (rvc-eval) is an appropriate name for this project as it highlights the simplified evaluation aspect of the original RVC system.

## Installation

1. Clone this repository:
```bash
git clone -b aug10version https://github.com/artificialnouveau/rvc-eval.git
```

2. Initialize and update the RVC submodule in the `rvc` directory:
```bash
cd rvc-eval
git submodule update --init --recursive
```

3. Install dependencies using Pip. rvc-eval only runs on python 3.10 so we need to create a separate conda environment
```bash
conda create -n rvc-eval python=3.10 --file requirements.txt
conda activate rvc-eval
pip install git+https://github.com/artificialnouveau/my-voice-analysis

pip install -e .

```

if pip install requirements fails, do the following
```bash
pip install openai-whisper python-osc
conda install -c conda-forge numpy=1.24 ffmpeg pyworld
```

4. Download the Hubert model (`hubert_base.pt`) from [Hugging Face](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) and place it in the `models` directory:

5. Copy the file myspsolution.praat from

                                      https://github.com/Shahabks/my-voice-analysis  

and save in the directory where you will save audio files for analysis.

Audio files must be in *.wav format, recorded at 44 kHz sample frame and 16 bits of resolution.

If you get an error like this: 

```OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized"```

then run this

```
export KMP_DUPLICATE_LIB_OK=TRUE
```
## Usage

To run the voice conversion system (with the default `hubert_base.pt` model or specify a custom path with the `--hubert` option):

1. To turn on the OSC receiver
```bash
python -m rvc_eval.cli --use-osc

# Do this if you want to use the analyze function (see more information below)
python -m rvc_eval.cli --use-osc --analyze
```

2. If you want to run from the command line:
```
python -m rvc_eval.cli --model path/to/your/models/models.pth --input-file path/to/your/input.wav --output-file path/to/your/output.wav
```

3. If you want to use RVC v1 or v2 just add --rvcversion:
If you do not specify the version, by default the rvcversion is "v2"
```
python -m rvc_eval.cli --model path/to/your/models/models.pth --input-file path/to/your/input.wav --output-file path/to/your/output.wav --rvcrevsion "v1"
```

4. If you want to include the speech analysis just add --analyze:
```
python -m rvc_eval.cli --model path/to/your/models/models.pth --input-file path/to/your/input.wav --output-file path/to/your/output.wav --analyze
```

## Analyze Function
The --analyze function allows for the analysis of audio files, primarily focusing on extracting voice features. It combines the capabilities of OpenAI's Whisper ASR model and the my-voice-analysis library to offer insights into voice recordings.
Features

    Transcribes audio content to text.
    Provides insights such as:
        Gender
        Emotion
        Number of syllables
        Number of pauses
        Rate of speech
        Articulation rate
        Speaking duration vs. original duration
        Fundamental frequency details: mean, std, median, min, max, 25th quantile, and 75th quantile.

All analysis results are saved in a JSON file for easy access and further processing.

## Recommended Dependencies and Requirements
- Python: 3.10.x
- PyTorch: 2.0.0+cu118
- Pipenv is used for managing dependencies.

## Credits
- This project is based on the [Retrieval-based Voice Conversion](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI) system by liujing04.
- This project also refers to [Voice Changer](https://github.com/w-okada/voice-changer) by w-okada, and some parts of the code are based on it.

## License

This project is licensed under the MIT License, following the licenses of the [original RVC repository](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI) and the [Voice Changer repository](https://github.com/w-okada/voice-changer). See the [LICENSE](LICENSE) file for details.

