from pathlib import Path

def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 10,
        "lr": 1e-4,
        "seq_len": 400,
        "d_model": 512,
        "datasource": "cfilt/iitb-english-hindi",        # or opusbooks
        "lang_src": "en",
        "lang_tgt": "hi",
        "tokenizer_file": "tokenizer_{}.json",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "experiment_name": "runs/eng-hi"
    }




def get_weights_file_path(config, epoch:str):
    model_folder= config['model_folder']
    model_basename= config['model_filename']
    model_filename= f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/ model_filename)
