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


def load_partial_state_dict(model, state_dict):
    """
    Load weights from the checkpoint into the model.
    Skip weights where the size doesn't match.
    """
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def load_net_g(model_path: str, is_half: bool, device: torch.device, version: str):
    from rvc.infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono, SynthesizerTrnMs768NSFsid, SynthesizerTrnMs768NSFsid_nono
    
    print("loading pth %s" % model_path)
    cpt = torch.load(model_path, map_location="cpu")
    print("rvc version: ", version)
    
    if version == "v1":
        sampling_rate = cpt["config"][-1]
        if_f0 = cpt.get("f0", 1)
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half).to(device)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"]).to(device)
    elif version == "v2":
        sampling_rate = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        if_f0 = cpt.get("f0", 1)
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half).to(device)
            net_g.eval()
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"]).to(device)
            net_g.eval()
        try:
            # Try to load all the weights first
            net_g.load_state_dict(cpt["weight"], strict=False)
        except RuntimeError as e:
            print("Encountered an error when loading weights. Attempting partial loading...")
            # If there's an error, try the partial loading
            load_partial_state_dict(net_g, cpt["weight"])
    
    # Register hook to print tensor shape for net_g
    net_g.register_forward_hook(print_shape_hook)
    
    net_g.eval()
    
    return (net_g.half() if is_half else net_g.float(), sampling_rate)
