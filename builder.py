import os
import shutil
import glob
import sys
import time
import warnings

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.multiprocessing as mp
import streamlit as st
import tqdm

# torchaudio currently (0.7) will throw warning that cannot be disabled
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import torchaudio

import simpleutils
from modules.emb_model import FpNetwork
from datautil.melspec import build_mel_spec_layer
from datautil.musicdata import MusicDataset

def build_vector_db(audios_dir_for_db: str, dir_for_db: str, configs: str, extensions: list):

    if not os.path.isdir(audios_dir_for_db):
        raise OSError(f"Directory {audios_dir_for_db} not found")
    
    file_list = []
    for ext in extensions:
        file_list += glob.glob(os.path.join(audios_dir_for_db, f"*.{ext}"))
    logger_init = simpleutils.MultiProcessInitLogger('builder')
    logger_init()
    
    if os.path.isdir(configs):
        configs_path = os.path.join(configs, 'configs.json')
        params = simpleutils.read_config(configs_path)
        params['model_dir'] = configs
        configs = configs_path
    else:
        params = simpleutils.read_config(configs)

    d = params['model']['d']
    h = params['model']['h']
    u = params['model']['u']
    F_bin = params['n_mels']
    segn = int(params['segment_size'] * params['sample_rate'])
    T = (segn + params['stft_hop'] - 1) // params['stft_hop']

    print('loading model...')
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    model = FpNetwork(d, h, u, F_bin, T, params['model']).to(device)
    model.load_state_dict(torch.load(os.path.join(params['model_dir'], 'model.pt'), map_location=device))
    print('model loaded')

    # doing inference, turn off gradient
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    params['indexer']['frame_shift_mul'] = 1
    dataset = MusicDataset(file_list, params)
    loader = DataLoader(dataset, num_workers=4, batch_size=None, worker_init_fn=logger_init)
    
    mel = build_mel_spec_layer(params).to(device)
    
    os.makedirs(dir_for_db, exist_ok=True)
    embeddings_file = open(os.path.join(dir_for_db, 'embeddings'), 'wb')
    lbl = []
    landmarkKey = []
    embeddings = 0

    progress_bar = st.progress(0.0)
    progress = 0.0
    status = st.empty()
    status.text(f"Progress: {progress}%")
    for i, dat in enumerate(tqdm.tqdm(loader)):
        progress = (i + 1) / len(loader.dataset)
        progress_bar.progress(progress)
        status.text(f"Progress: {progress*100 :.2f}%")

        logger = mp.get_logger()
        i, name, wav = dat
        logger.info('get music %s', name)
        tm_0 = time.time()
        i = int(i) # i is leaking file handles!
        
        if wav.shape[0] == 0:
            # load file error!
            print('load %s error!' % name)
            landmarkKey.append(0)
            continue
        
        for batch in torch.split(wav, 32):
            g = batch.to(device)
            
            # Mel spectrogram
            with warnings.catch_warnings():
                # torchaudio is still using deprecated function torch.rfft
                warnings.simplefilter("ignore")
                g = mel(g)
            z = model(g).cpu()
            for _ in z:
                lbl.append(i)
            embeddings_file.write(z.numpy().tobytes())
            embeddings += z.shape[0]
        landmarkKey.append(int(wav.shape[0]))
        tm_1 = time.time()
        logger.info('compute embedding %.6fs', tm_1 - tm_0)
    embeddings_file.flush()

    if embeddings == 0:
        print('The database is empty!')
        status.text("No embeddings have been created, database is empty")
    else:
        print('total', embeddings, 'embeddings')
        status.text(f"100% Done! {embeddings} embeddings have been created")
    #writer = tensorboardX.SummaryWriter()
    #writer.add_embedding(embeddings, lbl)
    
    # train indexer
    print('training indexer')
    try:
        index = faiss.index_factory(d, params['indexer']['index_factory'], faiss.METRIC_INNER_PRODUCT)
    except RuntimeError as x:
        if 'not implemented for inner prod search' in str(x) or "Error: 'metric == METRIC_L2' failed" in str(x):
            print(x)
            index = faiss.index_factory(d, params['indexer']['index_factory'], faiss.METRIC_L2)
        else:
            raise
    
    embeddings = np.fromfile(os.path.join(dir_for_db, 'embeddings'), dtype=np.float32).reshape([-1, d])
    if not index.is_trained:
        index.verbose = True
        try:
            index.train(embeddings)
        except RuntimeError as x:
            print(x)
            if "Error: 'nx >= k' failed" in str(x):
                index = faiss.IndexFlatIP(d)
    #index = faiss.IndexFlatIP(d)
    
    # write database
    print('writing database')
    index.add(embeddings)
    faiss.write_index(index, os.path.join(dir_for_db, 'landmarkValue'))
    
    landmarkKey = np.array(landmarkKey, dtype=np.int32)
    landmarkKey.tofile(os.path.join(dir_for_db, 'landmarkKey'))
    
    with open(os.path.join(dir_for_db, 'songList.txt'), 'w') as f:
        for i in file_list:
            f.write(f"{i}\n")
    f.close()
    
    # write settings
    shutil.copyfile(configs, os.path.join(dir_for_db, 'configs.json'))
    
    # write model
    shutil.copyfile(os.path.join(params['model_dir'], 'model.pt'), os.path.join(dir_for_db, 'model.pt'))
