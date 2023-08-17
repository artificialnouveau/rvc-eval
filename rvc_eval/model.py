import torch
from fairseq import checkpoint_utils

def print_shape_hook(module, input, output):
    print(f"{module.__class__.__name__} output shape: {output.shape}")


def load_hubert(model_path: str, is_half: bool, device: torch.device):
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [model_path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)

    # Register hook to print tensor shape for hubert_model
    hubert_model.register_forward_hook(print_shape_hook)

    hubert_model.eval()
    return hubert_model.to(device).half() if is_half else hubert_model.to(device).float()


def load_net_g(model_path: str, is_half: bool, device: torch.device, version: str):
    from rvc.infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono, SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
    cpt = torch.load(model_path, map_location="cpu")
    sampling_rate = cpt["config"][-1]
    if version == "v1":
        net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half).to(device)
    elif version == "v2":
        net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"], is_half=is_half).to(device)
        
    # Register hook to print tensor shape for net_g
    net_g.register_forward_hook(print_shape_hook)

    net_g.eval()
    net_g.load_state_dict(cpt["weight"], strict=False)
    return (net_g.half() if is_half else net_g.float(), sampling_rate)
