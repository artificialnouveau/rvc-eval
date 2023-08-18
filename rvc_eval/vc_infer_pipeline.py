# Original Code From: w-okada, liujing04
# Modified by: esnya

import numpy as np
import parselmouth
import pyworld
import scipy.signal as signal
import torch
import torch.nn.functional as F
from scipy.signal import resample
from rvc_eval.config import Config
from scipy.signal import resample_poly

class VC(object):
    def __init__(self, tgt_sr, device, is_half, x_pad, version):
        config = Config.get(is_half)
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.x_query = config.x_query
        self.x_pad = x_pad
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值
        self.device = device
        self.is_half = is_half
        self.version = version

    def _pm(
        self,
        x: np.ndarray,
        p_len: int,
        time_step: float,
        f0_min: float,
        f0_max: float,
    ):
        f0 = (
            parselmouth.Sound(x, self.sr)
            .to_pitch_ac(
                time_step=time_step / 1000,
                voicing_threshold=0.6,
                pitch_floor=f0_min,
                pitch_ceiling=f0_max,
            )
            .selected_array["frequency"]
        )
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        return f0

    def _hervest(self, x: np.ndarray, f0_max: float):
        f0, t = pyworld.harvest(
            x.astype(np.double),
            fs=self.sr,
            f0_ceil=f0_max,
            frame_period=10,
        )
        f0 = pyworld.stonemask(x.astype(np.double), f0, t, self.sr)
        f0 = signal.medfilt(f0, 3)
        return f0

    def get_f0(self, x, p_len, f0_up_key, f0_method):
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        f0 = (
            self._pm(x, p_len, time_step, f0_min, f0_max)
            if f0_method == "pm"
            else self._hervest(x, f0_max)
        )
        f0 *= pow(2, f0_up_key / 12)

        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak  # 1-0

    @torch.no_grad()
    def vc(
        self,
        model,
        net_g,
        sid,
        audio0: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
    ):
        # Process the audio tensor
        feats = audio0.half() if self.is_half else audio0.float()
        

        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False).to(self.device)

        # Extract features based on model version
        if self.version == "v1":
            output_layer = 9
        elif self.version == "v2":
            output_layer = 12
        else:
            raise ValueError(f"Invalid model version: {self.version}")

        with torch.no_grad():
            logits = model.extract_features(
                source=feats.to(self.device),
                padding_mask=padding_mask,
                output_layer=output_layer
            )
            feats = model.final_proj(logits[0]) if self.version == "v1" else logits[0]

        # Handle pitch adjustments
        if pitch is not None and pitchf is not None:
            original_feats = feats.clone()
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            original_feats = F.interpolate(original_feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            
            p_len = audio0.shape[0] // self.window
            if feats.shape[1] < p_len:
                p_len = feats.shape[1]
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
                
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf <= 0] = 0.5  # replace with desired value if needed
            pitchff = pitchff.unsqueeze(-1)
            
            feats = feats * pitchff + original_feats * (1 - pitchff)
            feats = feats.to(original_feats.dtype)

        # Generate audio
        p_len = torch.tensor([feats.shape[1]], device=self.device).long()
        with torch.no_grad():
            audio1 = (
                net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0]
            ).data

        return audio1


    def pad_or_trim_tensor(self, tensor, target_length):
        tensor_length = tensor.shape[-1]
        
        if tensor_length < target_length:
            padding_size = target_length - tensor_length
            padded_tensor = F.pad(tensor, (0, padding_size), mode="constant", value=0)
            return padded_tensor
        
        elif tensor_length > target_length:
            trimmed_tensor = tensor[..., :target_length]
            return trimmed_tensor

        return tensor



    def determine_target_length_v2(self, audio_tensor):
        target_length = audio_tensor.shape[0]
        return target_length


    def adjust_tensor_for_v2(self,audio_tensor, target_length):
        return F.interpolate(audio_tensor.unsqueeze(0).unsqueeze(0), size=target_length).squeeze(0).squeeze(0)


    def pipeline_v1(
        self,
        model,
        net_g,
        sid,
        audio,
        f0_up_key,
        f0_method,
    ):
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")

        p_len = audio_pad.shape[0] // self.window

        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = self.get_f0(audio_pad, p_len, f0_up_key, f0_method)
        pitch = torch.tensor(
            pitch[:p_len], device=self.device, dtype=torch.long
        ).unsqueeze(0)
        pitchf = torch.tensor(
            pitchf[:p_len],
            device=self.device,
            dtype=torch.float16 if self.is_half else torch.float32,
        ).unsqueeze(0)

        vc_output = self.vc(
            model,
            net_g,
            sid,
            torch.from_numpy(audio_pad),
            pitch,
            pitchf,
        )

        audio_output = (
            vc_output
            if self.t_pad_tgt == 0
            else vc_output[self.t_pad_tgt : -self.t_pad_tgt]
        )

        return audio_output


    def pipeline_v2(self, model, net_g, sid, audio, f0_up_key, f0_method):
        original_length = len(audio)

        # Resample from original rate to 16kHz
        num_samples_16k = int(original_length * 16000 / 44100)
        audio_16k = resample(audio, num_samples_16k)

        # Adjust padding based on model version
        target_length = len(audio_16k) + 2 * self.t_pad
        audio_tensor = torch.from_numpy(audio_16k)
        if len(audio_tensor) < target_length:
            padding_size = target_length - len(audio_tensor)
            audio_pad = F.pad(audio_tensor, (padding_size // 2, padding_size - (padding_size // 2)))
            audio_pad = audio_pad.numpy()
        else:
            audio_pad = audio_tensor.numpy()

        p_len = len(audio_pad) // self.window

        # Extract features
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = self.get_f0(audio_pad, p_len, f0_up_key, f0_method)
        pitch = torch.tensor(pitch[:p_len], device=self.device, dtype=torch.long).unsqueeze(0)
        pitchf = torch.tensor(
            pitchf[:p_len],
            device=self.device,
            dtype=torch.float16 if self.is_half else torch.float32,
        ).unsqueeze(0)

        # Voice conversion
        vc_output = self.vc(
            model,
            net_g,
            sid,
            torch.from_numpy(audio_pad),
            pitch,
            pitchf,
        )

        # Trimming padding from the vc_output if needed
        audio_output = (
            vc_output
            if self.t_pad_tgt == 0
            else vc_output[self.t_pad_tgt : -self.t_pad_tgt]
        )

        return audio_output
