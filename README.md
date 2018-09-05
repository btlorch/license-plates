# Forensic Reconstruction of Severely Degraded License Plates


## Getting Started

Tested on Ubuntu 16.04 with Python 3.5, TensorFlow 1.4.0

### Installation

Clone this repo.
```bash
git clone https://github.com/btlorch/license-plates.git
cd license-plates
```

Inside your virtual environment install required packages.
```bash
pip install -e requirements.txt
```

Download trained model to a directory of choice.

### Running the demo
```bash
cd src
jupyter notebook
```

Then open `demo.ipynb`.

Depending on the location of the trained weights, you may need to update the path in the first cell.

### Examples
![North Carolina license plate](assets/north_carolina_example.png "North Carolina license plate")

![Arizona license plate](assets/arizona_example.png "Arizona license plate")

![Vermont license plate](assets/vermont_example.png "Vermont license plate")