# Monty for Ultrasound POC

Demo code for Monty on Ultrasound data. Project produced during TBP Robot Hackathon 2025.

## Installation

The environment for this project is managed with [conda](https://www.anaconda.com/download/success).

To create the environment, run:

### ARM64 (Apple Silicon) (zsh shell)
```
conda env create -f environment.yml --subdir=osx-64
conda init zsh
conda activate ultrasound_perception
conda config --env --set subdir osx-64
```

### ARM64 (Apple Silicon) (bash shell)
```
conda env create -f environment.yml --subdir=osx-64
conda init
conda activate ultrasound_perception
conda config --env --set subdir osx-64
```

### Intel (zsh shell)
```
conda env create -f environment.yml
conda init zsh
conda activate ultrasound_perception
```

### Intel (bash shell)
```
conda env create -f environment.yml
conda init
conda activate ultrasound_perception
```

## Experiments

Experiments are defined in the `configs` directory.

After installing the environment, to run an experiment, run:

```bash
python run.py -e <experiment_name>
```

### Offline Experiments

To run training on an offline ultrasound dataset (.json files) run:
```bash
python run.py -e json_dataset_ultrasound_learning
```

Make sure the `data_path` in `env_init_args` of the `json_dataset_ultrasound_learning` config points to your dataset.

TODO: Add instructions for downloading dataset.

To run inference on an offline ultrasound dataset (.json files) run:
```bash
python run.py -e json_dataset_ultrasound_experiment
```
(again, making sure the `data_path` points to your dataset)

### Online Experiments

You will need to follow a series of steps to run live, online experiments. These
steps will setup both the ability to capture ultrasound images via the iPad app, as well
as track the position of the probe.

The basic commands to actually run experiments are:

For an interactive, live Ultrasound experiment (e.g., to evaluate inference during a
demo), run:
```bash
python run.py -e probe_triggered_experiment
```

For a live Ultrasound experiment to collect a new .json dataset (e.g., to create
a new training dataset), run:
```bash
python run.py -e probe_triggered_data_collection_experiment
```

However, before you run these, you will need to setup the iPad app with the ultrasound
probe, as well as the Windows machine and associated Vive Tracker to capture the
live position of the probe.

Instructions for the iPad app setup can be found [in the following README](./scripts/ipad_app/README.md).

Instructions for the tracker server can be found [in the following README](./scripts/htc_vive/README.md).

Once you have the above systems set up, follow the checklist below to begin an experiment:

#### First Object

To perform inference with the first object

- Get all equipment powered on and working
  - iPad: unlock
  - Vive USB dongle: plug into Windows computer
  - Base stations: power on and position either side of the phantom setup; the exact position does not matter, but you want to maximize their ability to see the probe when it is moving
  - Tracker: power on, and ensure visible to base stations
  - Ultrasound probe: plug into iPad
- Launch SteamVR on Windows
- Check probe orientation:

Ensure the tracker is attached to the probe, and matches the following orientation (two "legs" towards front of probe):

<img src="./custom_classes/figures/tracker_orientation.png" width="200"/>


You should also position the strap so that the center of the butterfly logo is approximately in the center of the gap in the strap, along the long axis of the probe. This will correspond to the center of the tracker puck relative to the probe tip being: ~8.4cm in the long axis and ~2.9cm in the short axis of the probe. If you position the tracker differently, you will need to measure the offset between the tracker and probe tip and adjust the ultrasound sensor relative to the agent position in the `get_state` function of the `ProbeTriggeredUltrasoundEnvironment` class accordingly.


- Start the custom Butterfly app on the iPad
- Run the tracker server on the Windows machine (`scripts/htc_vive/server.py`)
- Run the visualization script on the Mac machine (`scripts/visualize_probe/server.py`)
- Using the visualization that pops up, position the probe at the base of the phantom bag as shown below, the click the calibration button


<img src="./custom_classes/figures/tracker_relative_bag.png" width="200"/>

You should see this change reflected in the visualization service. Note this visualization is only for the operator's benefit, and to enable interpreting "goal-states" sent by Monty; it does not affect the measured locations or displacements within Monty as the probe moves.

- Run the Monty experiment `python run.py -e probe_triggered_experiment`
- In the iPad app, click `Start Imaging`
- Infer by moving the probe and capturing more images!

#### Next Object

To continue inference with a subsequent object

- Change the object in the bag
- Stop the Monty experiment (if it hasn't terminated on it's own already) + start it again with the above command
- Press the "Reset step counter" button in the iPad app
- Click two captures → Infer!



## Learn More!

You can find more information on how we customized Monty for this usecase in the writeup [here](./custom_classes/How_Monty_is_Customized.md).

TODO: Link to our docs showcase
TODO: Link to YouTube videos

## Development

After installing the environment, you can run the following commands to check your code.

### Run formatter

```bash
ruff format
```

### Run style checks

```bash
ruff check
```

### Run dependency checks

```bash
deptry .
```

### Run static type checks

```bash
mypy .
```

### Run tests

```bash
pytest
```
