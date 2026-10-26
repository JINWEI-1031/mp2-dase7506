"""Dump one scheme's choices to JSON in a fresh process.

Running each scheme in its own process mirrors the official protocol and avoids
`features.py` module-name collisions between scheme folders.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from harness import load_split  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder', required=True, help='scheme folder name under solutions/')
    ap.add_argument('--split', default='validation')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()

    root = HERE.parents[1]
    folder = root / 'solutions' / args.folder
    bank, arrays, gold, refs = load_split(args.split)

    method_path = folder / 'method.py'
    sys.path.insert(0, str(method_path.parent))
    spec = importlib.util.spec_from_file_location('scheme_method', method_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    ckpt = folder / 'artifacts' if (folder / 'artifacts').exists() else None
    choices = list(module.predict(bank, arrays, ckpt))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'folder': args.folder, 'split': args.split, 'choices': choices}) + '\n')
    print(f'{args.folder}: {len(choices)} choices -> {args.out}')


if __name__ == '__main__':
    main()
