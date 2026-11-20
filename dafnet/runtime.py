"""Configuration and command adapters; research implementations remain separate."""
import importlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value):
    if value is None:
        return None
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else ROOT / path)


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config['dataset'] not in ('uieb', 'euvp', 'lsui'):
        raise ValueError('dataset must be uieb, euvp, or lsui')
    config['checkpoint'] = resolve_path(config['checkpoint'])
    for key, value in config['training'].items():
        if key.endswith(('_dir', '_path')):
            config['training'][key] = resolve_path(value)
    for key in ('input_dir', 'output_dir'):
        config['stage01'][key] = resolve_path(config['stage01'][key])
    for key in ('TEST_INPUT_DIR', 'TEST_LABEL_DIR', 'SAVE_PATH', 'image_dir'):
        config['evaluation'][key] = resolve_path(config['evaluation'][key])
    return config


def require_directory(path):
    if not Path(path).is_dir():
        raise FileNotFoundError(f'Dataset directory not found: {path}. Edit the YAML paths.')


def precision_config(config):
    return {key: config['training']['precision_' + key] for key in
            ('shared_stem', 'expert_backbone', 'encoder', 'sfm', 'recon')}


def load_model(config, device):
    import torch
    from dafnet.model import PDSMPNetQuantLight64Enhanced
    model = PDSMPNetQuantLight64Enhanced(feat_ch=64, precision_cfg=precision_config(config)).to(device)
    state = torch.load(config['checkpoint'], map_location='cpu', weights_only=False)
    state = state['model_state'] if 'model_state' in state else state
    model.load_state_dict(state, strict=True)
    return model.eval()


def train(config):
    import sys
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    module = importlib.import_module('trainers.' + config['dataset'])
    settings = dict(config['training'])
    for key in ('input_dir', 'label_dir', 'test_input_dir', 'test_label_dir'):
        require_directory(settings[key])
    for key in ('save_dir', 'result_dir'):
        Path(settings[key]).mkdir(parents=True, exist_ok=True)
    settings['log_file'] = str(Path(settings['save_dir']) / 'log.txt')
    for key, filename in {'latest_checkpoint_path': 'latest_checkpoint.pth',
                          'interrupt_checkpoint_path': 'interrupt_checkpoint.pth',
                          'best_ema_path': 'best_model_ema.pth',
                          'final_ema_path': 'final_model_ema.pth'}.items():
        if config['dataset'] == 'lsui':
            settings[key] = str(Path(settings['save_dir']) / filename)
    for key, value in settings.items():
        setattr(module.Config, key, value)
    module.main()


def evaluation_module(config, initialize_metrics=False):
    module = importlib.import_module('dafnet.evaluation.' + config['dataset'])
    for key, value in config['evaluation'].items():
        setattr(module, key, value)
    module.WEIGHT_PATH = config['checkpoint']
    if initialize_metrics:
        module.niqe_metric = module.pyiqa.create_metric('niqe', device=module.DEVICE)
    return module


def enhance(config, input_path, output_dir):
    import torch
    from PIL import Image
    from torchvision import transforms
    from dafnet.utils import save_image, torch_to_np
    module = evaluation_module(config)
    model = load_model(config, module.DEVICE)
    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    extensions = getattr(module, 'IMG_EXTS', ('.png', '.jpg', '.jpeg'))
    files = [source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in extensions)
    if not files:
        raise ValueError(f'No supported images in {source}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    to_tensor = transforms.ToTensor()
    with torch.no_grad():
        for path in files:
            raw = Image.open(path).convert('RGB')
            image = torch.cat([to_tensor(raw), to_tensor(module.clahe_enhance(raw)),
                               to_tensor(module.gray_world_balance(raw))], dim=0).unsqueeze(0).to(module.DEVICE)
            output = model(image)['enhanced'].clamp(0, 1)
            save_image(path.name + '.png', torch_to_np(output), str(Path(output_dir)) + '/')
    return len(files)


def evaluate(config, efficiency=False, save_images=False):
    for key in ('TEST_INPUT_DIR', 'TEST_LABEL_DIR'):
        require_directory(config['evaluation'][key])
    module = evaluation_module(config, initialize_metrics=True)
    Path(module.SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    if efficiency:
        if config['dataset'] != 'uieb' and module.DEVICE.type != 'cuda':
            raise RuntimeError('Original EUVP/LSUI FPS measurement requires CUDA. Use test.py for quality metrics.')
        module.main()
    else:
        model = load_model(config, module.DEVICE)
        metrics = module.evaluate_quality(model)
        report = {'dataset': config['dataset'], 'checkpoint': config['checkpoint'],
                  'metrics': {key: float(value) for key, value in metrics.items()}}
        Path(module.SAVE_PATH).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    if save_images:
        count = enhance(config, module.TEST_INPUT_DIR, config['evaluation']['image_dir'])
        print(f'Saved {count} enhanced images to {config["evaluation"]["image_dir"]}')
