import io
import julius
import torch
import torchaudio
import numpy as np
import pandas as pd
import torch.nn.functional as F
import multiprocessing as mp
import time
from datetime import timedelta
from datautil.audio import stream_audio, WaveStream
import simpleutils

class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, params):
        self.params = params
        self.sample_rate = self.params['sample_rate']
        self.segment_size = int(self.sample_rate * self.params['segment_size'])
        self.hop_size = int(self.sample_rate * self.params['hop_size'])
        self.frame_shift_mul = self.params['indexer'].get('frame_shift_mul', 1)
        if isinstance(file_list, list):
            self.files = file_list
        else:
            self.files = simpleutils.read_file_list(file_list)
    
    def unsafe_getitem(self, index):
        logger = mp.get_logger()
        logger.info('MusicDataset getitem %s', self.files[index])
        tm_0 = time.time()
        smprate = self.sample_rate
        
        # resample
        stm = stream_audio(self.files[index])
        resampler = julius.ResampleFrac(stm.sample_rate, smprate)
        arr = []
        n = 0
        total = 0
        minute = stm.sample_rate * 60
        second = stm.sample_rate
        new_min = smprate * 60
        new_sec = smprate
        strip_head = 0
        wav = []

        tm_1 = time.time()
        tm_resample = 0.0
        tm_load = tm_1 - tm_0

        for b in stm.stream:
            tm_2 = time.time()
            tm_load += tm_2 - tm_1
            b = np.array(b).reshape([-1, stm.nchannels])
            b = np.multiply(b, 1/32768, dtype=np.float32)
            arr.append(b)
            n += b.shape[0]
            total += b.shape[0]
            if n >= minute:
                arr = np.concatenate(arr)
                b = arr[:minute]
                out = torch.from_numpy(b.T)
                wav.append(resampler(out)[:, strip_head : new_min-new_sec//2])
                arr = [arr[minute-second:].copy()]
                strip_head = new_sec//2
                n -= minute-second
            tm_1 = time.time()
            tm_resample += tm_1 - tm_2
        # resample tail part
        arr = np.concatenate(arr)
        out = torch.from_numpy(arr.T)
        wav.append(resampler(out)[:, strip_head : ])
        wav = torch.cat(wav, dim=1)

        tm_2 = time.time()
        tm_resample += tm_2 - tm_1
        logger.info('load %.6fs resample %.6fs', tm_load, tm_resample)
        
        # stereo to mono
        # check if it is fake stereo
        if wav.shape[0] == 2:
            pow1 = ((wav[0] - wav[1])**2).mean()
            pow2 = ((wav[0] + wav[1])**2).mean()
            if pow1 > pow2 * 1000:
                logger.warning('fake stereo with opposite phase detected: %s', self.files[index])
                wav[1] *= -1
        wav = wav.mean(dim=0)

        if wav.shape[0] < self.segment_size:
            # this "music" is too short and need to be extended
            wav = F.pad(wav, (0, self.segment_size - wav.shape[0]))
        
        # slice overlapping segments
        wav = wav.unfold(0, self.segment_size, self.hop_size//self.frame_shift_mul)
        wav = wav - wav.mean(dim=1).unsqueeze(1)

        tm_3 = time.time()
        logger.info('stereo to mono %.6fs', tm_3 - tm_2)
        
        return index, self.files[index], wav
        
    def __getitem__(self, index):
        try:
            return self.unsafe_getitem(index)
        except Exception as x:
            logger = mp.get_logger()
            logger.exception(x)
            return index, self.files[index], torch.zeros(0, self.segment_size)
    
    def __len__(self):
        return len(self.files)


class MusicDataset2(torch.utils.data.Dataset):
    def __init__(self, segments_csv, params):
        self.params = params
        self.sample_rate = self.params['sample_rate']
        self.segment_size = int(self.sample_rate * self.params['segment_size'])
        self.hop_size = int(self.sample_rate * self.params['hop_size'])
        self.frame_shift_mul = self.params['indexer'].get('frame_shift_mul', 1)
        self.segments = pd.read_csv(segments_csv)
        self.segments = self.segments[self.segments["class"] == "music"]

    
    def get_seconds(self, time_str):
        days_str, time_str = time_str.split(" days ")
        days = int(days_str)
        time_parts = time_str.split(':')
        hours, minutes, seconds = int(time_parts[0]), int(time_parts[1]), float(time_parts[2])
        time_delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
        return time_delta.total_seconds()
    
    def unsafe_getitem(self, index):
        filepath = self.segments.iloc[index]["path"]
        start = self.get_seconds(self.segments.iloc[index]["start"])
        end = self.get_seconds(self.segments.iloc[index]["end"])

        logger = mp.get_logger()
        logger.info('MusicDataset2 getitem %s', filepath)
        tm_0 = time.time()
        smprate = self.sample_rate

        sr = torchaudio.info(filepath).sample_rate
        wv, _ = torchaudio.load(filepath, frame_offset=int(start*sr), num_frames=int((end-start)*sr))
        audio_buffer = io.BytesIO()
        torchaudio.save(audio_buffer, wv, sr, format="wav", bits_per_sample=16)
        audio_buffer.seek(0)

        stm = WaveStream(file_buffer=audio_buffer, is_tmp=False)
        resampler = julius.ResampleFrac(stm.sample_rate, smprate)
        arr = []
        n = 0
        total = 0
        minute = stm.sample_rate * 60
        second = stm.sample_rate
        new_min = smprate * 60
        new_sec = smprate
        strip_head = 0
        wav = []

        tm_1 = time.time()
        tm_resample = 0.0
        tm_load = tm_1 - tm_0

        for b in stm.stream:
            tm_2 = time.time()
            tm_load += tm_2 - tm_1
            b = np.array(b).reshape([-1, stm.nchannels])
            b = np.multiply(b, 1/32768, dtype=np.float32)
            arr.append(b)
            n += b.shape[0]
            total += b.shape[0]
            if n >= minute:
                arr = np.concatenate(arr)
                b = arr[:minute]
                out = torch.from_numpy(b.T)
                wav.append(resampler(out)[:, strip_head : new_min-new_sec//2])
                arr = [arr[minute-second:].copy()]
                strip_head = new_sec//2
                n -= minute-second
            tm_1 = time.time()
            tm_resample += tm_1 - tm_2
        # resample tail part
        arr = np.concatenate(arr)
        out = torch.from_numpy(arr.T)
        wav.append(resampler(out)[:, strip_head : ])
        wav = torch.cat(wav, dim=1)

        tm_2 = time.time()
        tm_resample += tm_2 - tm_1
        logger.info('load %.6fs resample %.6fs', tm_load, tm_resample)
        
        # stereo to mono
        # check if it is fake stereo
        if wav.shape[0] == 2:
            pow1 = ((wav[0] - wav[1])**2).mean()
            pow2 = ((wav[0] + wav[1])**2).mean()
            if pow1 > pow2 * 1000:
                logger.warning('fake stereo with opposite phase detected: %s', filepath)
                wav[1] *= -1
        wav = wav.mean(dim=0)

        if wav.shape[0] < self.segment_size:
            # this "music" is too short and need to be extended
            wav = F.pad(wav, (0, self.segment_size - wav.shape[0]))
        
        # slice overlapping segments
        wav = wav.unfold(0, self.segment_size, self.hop_size//self.frame_shift_mul)
        wav = wav - wav.mean(dim=1).unsqueeze(1)

        tm_3 = time.time()
        logger.info('stereo to mono %.6fs', tm_3 - tm_2)
        
        return index, self.segments.iloc[index], wav
        
    def __getitem__(self, index):
        # try:
        #     return self.unsafe_getitem(index)
        # except Exception as x:
        #     logger = mp.get_logger()
        #     logger.exception(x)
        #     return index, self.segments.iloc[index], torch.zeros(0, self.segment_size)
        return self.unsafe_getitem(index)
    
    def __len__(self):
        return self.segments.shape[0]
