# Fine-tuned DistilBERT for Emotion Detection

This project demonstrates fine-tuning a DistilBERT model for emotion detection in text. The model achieves a significant accuracy improvement from around 25% to 90% on unseen data.

## Project Overview

This project aims to enhance the performance of a pre-trained DistilBERT model in emotion detection tasks. By fine-tuning it on an emotion dataset, the model can effectively classify text into different emotional categories.

## Dataset

The [five_emotions_data.csv](https://raw.githubusercontent.com/venkatareddykonasani/Datasets/master/Final_Emotion_Data/five_emotions_data.csv) dataset is used, which contains text labeled with five emotions: sadness, joy, love, anger, and fear.

## Methodology

1. **Data Preparation:** The dataset is downloaded, and the text data is preprocessed by cleaning and tokenizing it. The data is then split into training and testing sets.
2. **Model Fine-tuning:** The Hugging Face Transformers library is utilized to fine-tune a pre-trained DistilBERT model. The model is trained for a specific number of epochs, and the performance is evaluated on the test set.
3. **Evaluation:** After fine-tuning, the model is evaluated on unseen data to assess its performance in emotion detection.

## Results

The fine-tuned DistilBERT model achieves a remarkable 90% accuracy on unseen data for emotion detection, a significant improvement from the initial 25% accuracy before fine-tuning.

## Steps to Reproduce

1. **Install dependencies:**

2. **Download the dataset:**

3. **Fine-tune the DistilBERT model (refer to `FineTuned2(distil).ipynb` for detailed code):**

   - Load the dataset and preprocess it.
   - Load the pre-trained DistilBERT model and tokenizer.
   - Define training arguments and create a `Trainer` instance.
   - Train the model using the `trainer.train()` method.
   - Evaluate the model on the test set.

## Files

* `FineTuned2(distil).ipynb`: The main Jupyter notebook containing the code for fine-tuning and evaluation.
* `requirements.txt`: Lists the project dependencies.

## Acknowledgments

* Hugging Face Transformers library
* [venkatareddykonasani/Datasets](https://github.com/venkatareddykonasani/Datasets) for the emotions dataset

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.
