# Forensic Reconstruction of Severely Degraded License Plates


## Getting Started

Tested on Ubuntu 16.04 with Python 3.5, TensorFlow 1.4.0 and 1.10.1.

### Installation

Clone this repo.
```bash
git clone https://github.com/btlorch/license-plates.git
cd license-plates
```

Inside your virtual environment install required packages.
```bash
pip install -r requirements.txt
```

[Download trained model](https://faui1-files.cs.fau.de/public/mmsec/license-plates/license-plates-trained-model.zip) trained model to `<repo>/model` or a directory of your choice.
```bash
wget https://faui1-files.cs.fau.de/public/mmsec/license-plates/license-plates-trained-model.zip
unzip license-plates-trained-model.zip -d model
```

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

### CNN architecture

(View image to enlarge)

![CNN architecture](assets/cnn_architecture.png "CNN architecture")