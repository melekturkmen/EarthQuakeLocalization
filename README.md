# EarthQuake_localization
Python code demonstrating predictions and actual values of earthquake epicenters over Turkey.

## Introduction
This repository contains pre-trained model to make inferences and samples to plot the model predictions against actual earthquake locations.

By executing this file, runme.py, you can visualize the results. You can use the following command to run the code. 
```bash
python runme.py
```

## Limitations
This function operates only on 60-sec or longer events with ResNet-based architecture, using input signal in the frequency domain (i.e. STFT). The network architecture is shown below. Please refer to the arXiv paper (link?) for details. 

![image](https://github.com/melekturkmen/EarthQuake_localization/assets/44256504/e20ab4c6-1e19-4cc2-b2fd-5f66d0900cfc)


## Installation
No additional installations (other than PyTorch) are required to run this project. All requirements are basic.

## Usage
Clone the repository to your local machine.
Navigate to the project directory, and adjust the code according to the working directory.
Run the runme.py file using the command provided above.

## Output

The output figures for sample EQ events (sampleQuakes folder) are saved in "figs" folder. A sample result is shown below. 

![image](https://github.com/melekturkmen/EarthQuake_localization/assets/44256504/65dfe388-ab14-46fb-a538-4b3f3a2e2b1d)


## Contributions
Contributions are welcome. Feel free to provide any feedback or suggestions.


## Contact Information
Melek Türkmen

Middle East Technical University

Graduate School of Informatics

turkmen.melek@metu.edu.tr
