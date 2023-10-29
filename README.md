TAED2-Falsettos
==============================

A Data Science project that uses best MLOps practices to create an API with a model to predict speech commands. Check the:

- [Dataset Card](dataset-card.md) 
- [Model Card](model-card.md)

Project Structure
------------

    ├── .dvc
    │   ├── .gitignore    
    │   └── config    
    │
    ├── data
    │   ├── .gitignore
    │   ├── audio_examples.dvc
    │   ├── raw.dvc
    │   └── raw_sample_example.dvc 
    │
    ├── docs
    │   ├── Makefile.txt
    │   ├── commands.rst
    │   ├── conf.py
    │   ├── getting-started.rst
    │   ├── index.rst
    │   └── make.bat 
    │
    ├── gx
    │   ├── checkpoints
    │   │   └── my_checkpoint.yaml
    │   ├── expectations
    │   │   ├── .ge_store_backend_id
    │   │   └── speech_commands_suite.json
    │   ├── plugins
    │   │   └── custom_data_docs
    │   │       └── styles
    │   │           └── data_docs_custom_styles.css
    │   ├── .gitignore
    │   └── great_expectations.yaml 
    │
    ├── notebooks
    │   ├── audio_duration_distribution.ipynb
    │   ├── get_data_sample.ipynb
    │   ├── get_raw_data.ipynb
    │   ├── label_distribution.ipynb
    │   └── make_prediction_API.ipynb 
    │
    ├── src                
    │   ├── __init__.py   
    │   ├── app        
    │   │   ├── api.py
    │   │   └── schemas.py
    │   ├── data           
    │   │   ├── __init__.py
    │   │   └── preprocess_dataset.py
    │   ├── features      
    │   │   └── validate.py
    │   ├── models        
    │   │   ├── __init__.py
    │   │   ├── eval_model.py
    │   │   ├── Hubert_Classifier_model.py
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   └── tests
    │       ├── test_api.py
    │       └── test_model.py
    │
    ├── .gitignore
    │
    ├── .pylintrc
    │
    ├── __init__.py
    │
    ├── dataset-card.md
    │
    ├── dvc.lock
    │
    ├── dvc.yaml
    │
    ├── LICENSE
    │
    ├── model-card.md
    │
    ├── params.yaml
    │
    ├── README.md
    │
    ├── requirements.txt
    │
    └── setup.py          

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
