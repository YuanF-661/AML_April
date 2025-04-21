# AML_April

This repository contains code and resources for audio machine learning experiments conducted in April.  
It includes various models and data processing scripts aimed at audio classification and analysis tasks.

---

## Project Structure

- **MFCC-CNN-OneHot/**: Implementation of a Convolutional Neural Network (CNN) using Mel-Frequency Cepstral Coefficients (MFCC) features with one-hot encoded labels.
- **MFCC-CNN/**: CNN model utilizing MFCC features for audio classification.
- **PANNsTeachers/**: Code related to Pretrained Audio Neural Networks (PANNs) used as teacher models for knowledge distillation.
- **SimCLR/**: Implementation of the SimCLR framework for self-supervised learning on audio data.
- **drum_analysis_results/**: Results and analysis related to drum sound classification experiments.

---

## Data Processing Scripts

- **DataAugment_Librosa.py**: Applies data augmentation techniques using the Librosa library.
- **DataAugment_PedalBoardNew.py**: Applies data augmentation using the Pedalboard library.
- **FolderFileNumbers.py**: Utility script to count the number of files in dataset folders.
- **ProcessDataSliceRename.py**: Processes and renames sliced audio data for consistency.
- **RawDataSliceRename.py**: Handles slicing and renaming of raw audio data.
- **Sketch.py**: Contains experimental code snippets and testing routines.

---

## Getting Started

### Clone the repository

```bash
git clone https://github.com/YuanF-661/AML_April.git
cd AML_April
```

### Set up the environment

Ensure you have Python 3.7 or higher installed. Then install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note**: If `requirements.txt` is not present, manually install the necessary libraries such as `librosa`, `pedalboard`, `torch`, etc.

### Prepare the dataset

Place your audio dataset in the appropriate directory structure as expected by the scripts.  
Ensure that the data is preprocessed using the provided scripts before training the models.

### Train the models

Navigate to the desired model directory and follow the instructions in the respective README or script files to train the models.

---

## Contributing

Contributions are welcome!  
Please fork the repository and submit a pull request for any enhancements or bug fixes.

---

## License

This project is licensed under the MIT License.
