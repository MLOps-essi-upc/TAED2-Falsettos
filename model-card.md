# Model Card for Hubert_Classifier

The idea of the models in our case is to predict from a one-second .wav audio file (single spoken English word or background noise) whether it is noise or a word. So the task is Audio Classification and the sub-task is Keyword Spotting from non-noise audios. 

## Model Details

### Model Description

This model is a deep learning model designed for audio classification tasks. It is built upon the HubertModel, which is a pre-trained model specifically designed for speech and audio processing. The model incorporates an adapter and a classifier to adapt the HubertModel for audio classification.

1. HubertModel: This component is initialized with a pre-trained HubertModel. The HubertModel is responsible for extracting high-level representations from the input audio data.
2. Adaptor: The adaptor component is a feed-forward neural network that takes the output from the HubertModel and performs adaptation to better fit the audio classification task. It consists of two linear layers with ReLU activation and dropout in between. The first linear layer reduces the dimensionality of the hidden representations to a specified adapter_hidden_size, and the second linear layer
brings it back to the original hidden_size.
3. Classifier: The classifier component further processes the adapted representations for classification. It consists of two linear layers with ReLU activation and dropout in between. The first linear layer reduces the dimensionality to adapter_hidden_size, and the final linear layer maps it to a single output node, indicating the classification
prediction.

- **Developed by:** Armand de Asís, Sergio Cárdenas and Jan Sallent
- **Model type:** Transformer model
- **Language(s) (NLP):** English
- **License:** cc-by-4.0

### Model Sources

- **Repository:** https://huggingface.co/docs/transformers/model_doc/hubert
- **Library:** https://pypi.org/project/transformers/
- **Paper:** https://arxiv.org/abs/2106.07447

## Uses

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

Given an audio file (.wav) of one second the idea is to give a label of the word (English word) said or whether it's noise. 

### Downstream Use 

It could be tuned to be used to longer than 1 second wav audio files or other audio sources.

### Out-of-Scope Use

This model is not intended to be used in other languages or sounds. 


## Bias, Risks, and Limitations

We could have a bias towards a concrete language English dialect or to a concrete audio recording device. So when using the model in other cases, the error rate could increase and have a possible bias to noise or specific words. 

### Recommendations

Ensure a slow and clear pronunciation when recording the words. Avoid surrounding noises when recording the audios. 


## How to Get Started with the Model

Use the code below to get started with the model:

First import libraries needed (such as torch or transformers), also add the class HubertForAudioClassification. In predict_model.py you will find all the libraries needed and the class HubertForAudioClassification. 

To load the best model, first initialize the model using our class HubertForAudioClassification. Then get the path .pt or .pth archive where are saved the best weights of the model. Then, load the weights to the model:

```python
model = HubertForAudioClassification(adapter_hidden_size=params["model"]["adapter_hidden_size"])

PATH = os.path.join(MODELS_DIR,'final_model', '{}_bestmodel.pt'.format(params["model"]["algorithm_name"]))
    
model.load_state_dict(torch.load(PATH), strict=False)
```

Then, move the model to CPU or GPU (in this case we move it to CPU) and then set model to evaluate:

```python
model.cpu()
model.eval()
```

Finally, use the model to predict on a single audio file (even though many can be forwarded in the batch dimension). First, load the audio file. The entry is a tensor of floats of precision 32 bits. This was created using preprocessing pipeline, we recommend you to follow the same procedure to avoid a lower performance. Remember to add batch dim using unsqueeze(0) to the tensor. 

And then use the model to predict, we must use softmax (or logsoftmax to have better numerical stability) to get the probabilities of each class. Then, we get the index of the maximum value along the class dimension. This index is the predicted class. Finally, we get the word associated to the predicted class (using the dictionary of words provided by the dataset).

```python
with torch.no_grad():
        # Get the output
        logits = model(data_sample)
        output = F.log_softmax(logits, dim=1)
        # Get the indices of the maximum values along the class dimension
        predicted_class = torch.argmax(output, dim=1)
```
Using these commands we will know exactly the class predicted, and it's reference in the dictionary of words (provided by dataset).
```python
    label = predicted_class.int().item()
    print("Predicted label:", label)
    print("Predicted word:", UNKNOWN_WORDS_V2[label])
```

## Training Details

### Training Data
- [Dataset Card](dataset-card.md) 

|       | train | validation |
|:-----:|:-----:|:----------:|
| v0.01 | 51093 |    6799    |
| v0.02 | 84848 |    9982    |

This is a set of one-second .wav audio files, each containing a single spoken English word or background noise. These words are from a small set of commands, and are spoken by a variety of different speakers. This data set is designed to help train simple machine learning models. It is covered in more detail at https://arxiv.org/abs/1804.03209.

Data splits were given from the authors of the dataset. 

### Training Procedure 

#### Preprocessing 

We delete some columns that we don't need from raw dataset, and make some changes in labelling and audio. The audio is stored as a vector, so we ensure the length of the audio is exactly 1 second or 16000 samples (sample rate must be 16kHz). 


#### Training Hyperparameters

- **batch_size**: 256 # training and valid batch size
- **lr**: 0.00001 # learning rate
- **momentum**: 0.9 # SGD momentum, for SGD only
- **optimizer**: 'adam' # optimization method: sgd | adam
- **adapter_hidden_size**: 128 # model hyperparameter

- **epochs**: 100  # maximum number of epochs to train
- **patience**: 5 # how many epochs of no loss improvement should we wait before stop training
- **log_interval**: 15 # how many batches to wait before logging training status

#### Speeds, Sizes, Times

We make checkpoints each time an epoch gives better results compared to the last one. We save the best model in a .pt archive. All local archives are saved in the folder "models" in the root of the project but ignored from github. So MLflow and DVC only stores the best model found through all the epochs.

The throughput is in MLFLow and DVC, so you can see the results in the links provided in the README.md file.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data
- [Dataset Card](dataset-card.md) 

|       | test |
|:-----:|:----:|
| v0.01 | 3081 |
| v0.02 | 4890 |

#### Metrics

The metric we will use for evaluating the model is F1 score. We couldn’t use accuracy as the classes had imbalance, and this new metric is particularly useful because it focuses on the performance of the minority class. We can observe in this figure (generated by “label distribution.ipynb”) the number of entries per class. 

Also we record the Loss average per epoch and the used loss is the CrossEntropyLoss.

### Results

{{ results | default("[More Information Needed]", true)}}

#### Summary

{{ results_summary | default("", true) }}

## Environmental Impact

Carbon emissions are estimated using the codecarbon library for Python.

- **Hardware Type:** Desktop device (RTX 3060 + 32GB RAM + i5 12400F)
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** DagsHub (https://dagshub.com/) to experiment tracking and storing
- **Compute Region:** Spain
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}} kg CO2

## Model Card Contact
e-mail: armand.de.asis@estudiantat.upc.edu



