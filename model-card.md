# Model Card for HUBERT_00

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

Given an audio file (.wav) of one second the idea is to give a label of the word (English word) said or whether  it's noise. 

### Downstream Use 

It could be tuned to be used to longer than 1 second wav audio files or other audio sources.

### Out-of-Scope Use

This model is not intended to be used in other languages or sounds. 


## Bias, Risks, and Limitations

We could have a bias towards a concrete language English dialect or to a concrete audio recording device. So when using the model in other cases, the error rate could increase and have a possible bias to noise or specific words. 

### Recommendations

Ensure a slow and clear pronunciation when recording the words. Avoid surrounding noises when recording the audios. 


## How to Get Started with the Model

///TO BE DONE-------------------------------------------------------------------
Use the code below to get started with the model. (TO BE DONE)

## Training Details

### Training Data

<!-- This should link to a Data Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

{{ training_data | default("[More Information Needed]", true)}}

### Training Procedure 

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

{{ preprocessing | default("[More Information Needed]", true)}}


#### Training Hyperparameters

- **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Data Card if possible. -->

{{ testing_data | default("[More Information Needed]", true)}}

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

{{ testing_metrics | default("[More Information Needed]", true)}}

### Results

{{ results | default("[More Information Needed]", true)}}

#### Summary

{{ results_summary | default("", true) }}

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

{{ model_examination | default("[More Information Needed]", true)}}

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** {{ hardware | default("[More Information Needed]", true)}}
- **Hours used:** {{ hours_used | default("[More Information Needed]", true)}}
- **Cloud Provider:** {{ cloud_provider | default("[More Information Needed]", true)}}
- **Compute Region:** {{ cloud_region | default("[More Information Needed]", true)}}
- **Carbon Emitted:** {{ co2_emitted | default("[More Information Needed]", true)}}

## Technical Specifications [optional]

### Model Architecture and Objective

{{ model_specs | default("[More Information Needed]", true)}}

### Compute Infrastructure

{{ compute_infrastructure | default("[More Information Needed]", true)}}

#### Hardware

{{ hardware | default("[More Information Needed]", true)}}

#### Software

{{ software | default("[More Information Needed]", true)}}

///TO BE DONE-------------------------------------------------------------------

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

{{ citation_bibtex | default("[More Information Needed]", true)}}

**APA:**

{{ citation_apa | default("[More Information Needed]", true)}}

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

{{ glossary | default("[More Information Needed]", true)}}

## Model Card Contact

{{ model_card_contact | default("[More Information Needed]", true)}}



