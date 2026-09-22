# Smart Glasses

An early-stage computer-vision assistive-technology prototype for recognizing traffic signs and, later, providing spoken feedback to visually impaired users. This repository currently contains only the project structure; it does not download data, train a model, or run an application.

## Architecture

```text
src/smart_glasses/
├── capture/          Planned camera frame acquisition
├── detection/        Planned YOLO inference boundary
├── tracking/         Planned detection-to-track association
├── decision/         Planned policy for actionable events
├── interpretation/   Planned traffic-sign meaning and text
└── speech/           Planned text-to-speech output boundary

training/             Future training and evaluation entry points
data/                 Local datasets, organized by processing stage
models/               Local model weights and exported artifacts
tests/                Unit and integration tests
```

Keeping these planned responsibilities separate lets camera hardware, model choice, tracking strategy, interpretation rules, and speech provider evolve independently. When implementation begins, pretrained YOLO inference will belong in `detection`; custom-model training and evaluation will belong in `training`.

## Setup (macOS)

Python 3.10 or newer is recommended. A `.venv` already exists in this workspace. Activate it and install the declared dependencies:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If starting from a fresh clone, create the environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the active environment after installation:

```bash
python scripts/verify_environment.py
```

For local package imports during development, set the source directory on your Python path:

```bash
export PYTHONPATH="$PWD/src"
```

## Repository conventions

- Put original datasets in `data/raw`, transient transforms in `data/interim`, and training-ready material in `data/processed`.
- Put downloaded weights and exported models in `models`. Datasets, model weights, logs, and training outputs are intentionally excluded from Git.
- Keep production code in `src/smart_glasses`; do not put application logic in training scripts or notebooks.
- Add focused tests under `tests` as behavior is introduced.

## Files created in this scaffold

- `.gitignore` prevents machine-local, generated, dataset, model, and training artifacts from entering version control.
- `requirements.txt` declares the requested Python ML, vision, and test dependencies.
- `src/smart_glasses/__init__.py` establishes the importable application package.
- The six package modules document the boundaries for capture, detection, tracking, decision-making, sign interpretation, and speech output; they deliberately contain no implementation yet.
- `training/README.md` reserves a focused location for future train/evaluate workflows.
- `data/*/.gitkeep` and `models/.gitkeep` retain otherwise-empty local storage directories in Git without tracking their contents.
- `tests/README.md` reserves the testing location and records its intended scope.
