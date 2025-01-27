# Deep Learning-based Epicenter Localization using Single-Station Strong Motion Records
Python code applying deep learning techniques to strong motion records in order to locate epicenters at single stations of Türkiye. Please refer to the arXiv paper (https://arxiv.org/abs/2405.18451) for details. 
This study examines whether strong motion records, which are rarely used for seismology-related studies, contain information about an earthquake's characteristics, and whether DL-based methods can benefit from them.
The paper introduces a large-scale strong motion record collection, AFAD-1218, which contains over 36,000 strong motion records from Turkey.

## Introduction
This repository contains pre-trained model to make inferences and samples to plot the model predictions against actual earthquake locations. 

By executing this file, runme.py, you can visualize the results. You can use the following command to run the code. 
```bash
python runme.py
```

## Dataset

This study uses the **AFAD-1218 dataset**, a comprehensive collection of over 36,000 strong motion records from Turkey. These records were obtained from Turkey's national strong-motion network operated by the Disaster and Emergency Management Authority (AFAD). The dataset spans a wide range of seismic events and regions, providing a rich resource for deep learning applications.

### Characteristics of the AFAD-1218 dataset:
- **Number of Records:** 36,418  
- **Sampling Rate:** All sampled at 100 Hz  
- **Duration:** Varying event durations from 5 to 300 seconds  
- **SNR:** Ranging from a few dBs to 100 dB. Signals with SNR values lower than 25 dB were eliminated during the training process.  
- **Geographic Coverage:** Nation-wide strong ground motion stations across Turkey  
- **Features:** Includes ground acceleration time series and metadata, such as event magnitude, epicenter location, and station coordinates.  

The dataset was preprocessed to include only events with sufficient signal-to-noise ratio and to transform the raw time-domain signals into frequency-domain representations using the Short-Time Fourier Transform (STFT).

For more details, refer to our [arXiv publication on deep learning-based earthquake epicenter localization](https://arxiv.org/abs/2405.18451).

## Limitations
This repo operates only on 60-sec or longer events with ResNet-based architecture, using input signal in the frequency domain (i.e. STFT). The network architecture is shown below.

![image](https://github.com/melekturkmen/EarthQuake_localization/assets/44256504/3fc19cbb-86f2-440a-8723-e56a3b3f8084)

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
