# RVC Eval Simplified

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
git clone https://github.com/artificialnouveau/rvc-eval.git
```

2. Initialize and update the RVC submodule in the `rvc` directory:
```bash
cd rvc-eval
git submodule update --init --recursive
```

3. Install dependencies using Pipenv:
```bash
pip install -e .
```

4. Download the Hubert model (`hubert_base.pt`) from [Hugging Face](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) and place it in the `models` directory:


## Usage

1. To list available audio devices:
```bash
python -m rvc_eval.cli --list-audio-devices
```
If you want to run using OSC:
2. To run the voice conversion system (with the default `hubert_base.pt` model or specify a custom path with the `--hubert` option):
```
python -m rvc_eval.cli --model path/to/your/models/models.pth --input-file path/to/your/input.wav --output-file path/to/your/output.wav
```

If you want to run from the command line:
2. To run the voice conversion system (with the default `hubert_base.pt` model or specify a custom path with the `--hubert` option):
```
python -m rvc_eval.cli --model path/to/your/models/models.pth --input-file path/to/your/input.wav --output-file path/to/your/output.wav
```

## Recommended Dependencies and Requirements
- Python: 3.10.x
- PyTorch: 2.0.0+cu118
- Pipenv is used for managing dependencies.

## Credits
- This project is based on the [Retrieval-based Voice Conversion](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI) system by liujing04.
- This project also refers to [Voice Changer](https://github.com/w-okada/voice-changer) by w-okada, and some parts of the code are based on it.

## License

This project is licensed under the MIT License, following the licenses of the [original RVC repository](https://github.com/liujing04/Retrieval-based-Voice-Conversion-WebUI) and the [Voice Changer repository](https://github.com/w-okada/voice-changer). See the [LICENSE](LICENSE) file for details.

