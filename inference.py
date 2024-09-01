import logging
import os
import tqdm
import glob
import json
import asyncio
import argparse
import warnings
import torch
import torchaudio
import torchvision
import numpy as np
import pandas as pd
import streamlit as st
from datetime import timedelta
from modules import AudioDetectionNetwork
from train import load_config
from typing import *

warnings.filterwarnings(action="ignore")
logger = logging.getLogger(__name__)


def load_model_weights(model: AudioDetectionNetwork, model_path: str, device: str="cpu"):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"path: {model_path} does not exist")
    state_dict = torch.load(model_path, map_location=device)
    model_weights = state_dict["network_params"]
    model.to(device)
    model.init_zeros_taper_window(model_weights["taper_window"])
    model.load_state_dict(model_weights)
    model.eval()


def get_label_map(label_map_path: str) -> Dict[int, str]:
    with open(label_map_path, "r") as f:
        map_dict = json.load(f)
    f.close()
    map_dict = {int(k):v for k, v in map_dict.items()}
    return map_dict


def process_model_outputs(
        outputs: torch.Tensor,
        iou_threshold: float=0.05,
        conf_threshold: float=0.5,
        sample_duration: float=60,
        return_start_end: bool=True,
        _h: int=10,
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:

    if outputs.ndim != 3:
        outputs = outputs.unsqueeze(0)
    assert outputs.ndim == 3, "input is expected to have 2 or 3 dimensions"
    _device, _dtype = outputs.device, outputs.dtype
    cw = outputs[..., -2:]
    x1 = cw[..., :1] - (cw[..., -1:] / 2)
    x2 = cw[..., :1] + (cw[..., -1:] / 2)
    y1 = torch.zeros_like(x1, device=_device, dtype=_dtype)
    y2 = torch.zeros_like(x2, device=_device, dtype=_dtype) + _h
    coords = torch.cat([x1, y1, x2, y2], dim=-1).clip(min=0, max=sample_duration).squeeze(0)
    objectness = outputs[..., :1].sigmoid()
    class_scores = torch.nn.functional.softmax(outputs[..., 1:-2], dim=-1)
    class_scores = torch.gather(class_scores, dim=-1, index=class_scores.argmax(dim=-1, keepdim=True))
    confidence = class_scores * objectness
    batch_size = outputs.shape[0]
    num_preds_per_batch = outputs.shape[1]
    num_boxes = (batch_size * num_preds_per_batch)
    batch_idx = torch.zeros(num_boxes, device=_device, dtype=torch.int64)
    _idx = 0
    for i in range(0, num_boxes, num_preds_per_batch):
        batch_idx[i:i+num_preds_per_batch] = _idx
        _idx += 1
    flattened_coords = coords.flatten(0, -2)
    flattened_confidence = confidence.reshape(-1)
    keep = torchvision.ops.batched_nms(
        flattened_coords, 
        flattened_confidence.squeeze(), 
        idxs=batch_idx, 
        iou_threshold=iou_threshold
    )
    outputs = outputs.flatten(0, -2)
    kept_batch_idx = batch_idx[keep]
    kept_confidence = flattened_confidence[keep]
    kept_outputs = outputs[keep]
    valid_mask = kept_confidence > conf_threshold
    valid_batch_idx = kept_batch_idx[valid_mask]
    valid_confidence = kept_confidence[valid_mask]
    valid_outputs = kept_outputs[valid_mask]
    valid_outputs = torch.cat([valid_confidence.unsqueeze(-1), valid_outputs], dim=-1)
    
    batch_idx_list = []
    segments_list = []
    for i in torch.unique(valid_batch_idx).sort().values:
        segments = valid_outputs[valid_batch_idx == i]
        sort_idx =  segments[:, -2].argsort()
        segments = segments[sort_idx]
        segments[..., -2] = segments[..., -2] #+ (i * sample_duration)
        segments_list.append(segments)
        batch_idx_list.append(valid_batch_idx[valid_batch_idx == i])
    segments = torch.cat(segments_list, dim=0)
    batch_idxs = torch.cat(batch_idx_list, dim=0)
    if return_start_end:
        w = segments[..., -1]
        segments[..., -2] = segments[..., -2] - (w / 2)
        segments[..., -1] = segments[..., -2] + w
        segments[..., -2:] = segments[..., -2:].clip(min=0, max=sample_duration)

    class_labels = segments[..., 2:-2].argmax(dim=-1, keepdim=True)
    segments = torch.cat([segments[..., :2], class_labels, segments[..., -2:]], dim=-1)
    return segments, batch_idxs


def evaluate_audio(
        model: AudioDetectionNetwork, 
        audio_filepath: str, 
        output_dir: str,
        input_sample_rate: int, 
        sample_duration: float, 
        batch_size: int,
        idx2class_map: Dict[int, str],
        device: str="cpu",
        iou_threshold: float=0.1, 
        conf_threshold: float=0.65,
        save_result: bool=True,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:

    batch_start = 0
    batch_end = batch_size * sample_duration
    segments_list, batch_idxs_list = [], []
    _, og_sample_rate = torchaudio.load(audio_filepath, frame_offset=0, num_frames=10_000)
    sample_size = int(sample_duration * og_sample_rate)
    while True:
        batch_audio_tensor, _ = torchaudio.load(
            audio_filepath, 
            frame_offset=int(batch_start * og_sample_rate), 
            num_frames=int((batch_end - batch_start) * og_sample_rate),
            backend="soundfile"
        )
        if batch_audio_tensor.shape[-1] == 0: break

        batch_audio_tensor = batch_audio_tensor.squeeze(0)
        if batch_audio_tensor.ndim == 2 and batch_audio_tensor.shape[0] > 1:
            batch_audio_tensor = batch_audio_tensor.mean(dim=0).squeeze(0)

        nbatch = batch_size
        if batch_audio_tensor.shape[0] % sample_size != 0:
            nbatch = np.ceil(batch_audio_tensor.shape[0] / sample_size).astype(int)
            pad_size =  (nbatch * sample_size) - batch_audio_tensor.shape[0]
            _zeros = torch.zeros((pad_size, ), dtype=batch_audio_tensor.dtype)
            batch_audio_tensor = torch.cat([batch_audio_tensor, _zeros], dim=0)

        batch_audio_tensor = batch_audio_tensor.reshape(-1, 1, sample_size)
        batch_audio_tensor = batch_audio_tensor.to(device)

        with torch.no_grad():
            if og_sample_rate != input_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=og_sample_rate, 
                    new_freq=input_sample_rate
                )
                resampler.to(device)
                batch_audio_tensor = batch_audio_tensor.to(device)
                batch_audio_tensor = resampler(batch_audio_tensor)
            output: torch.Tensor = model(batch_audio_tensor, combine_scales=True)
        num_class_outputs =  output.shape[-1] - 3

        if num_class_outputs > len(idx2class_map):
            raise RuntimeError(f"model output does not match idx2class mapping")
        segments, batch_idxs = process_model_outputs(
            output, 
            iou_threshold=iou_threshold, 
            conf_threshold=conf_threshold, 
            sample_duration=sample_duration, 
            return_start_end=True
        )
        if len(batch_idxs_list) > 0:
            batch_idxs += batch_idxs_list[-1][-1]
        segments_list.append(segments)
        batch_idxs_list.append(batch_idxs)
        batch_start = batch_end
        batch_end += batch_size * sample_duration
    
    segments = torch.cat(segments_list, dim=0)
    batch_idxs = torch.cat(batch_idxs_list, dim=0)
    segments[..., -2:] = segments[..., -2:] + (batch_idxs.unsqueeze(-1) * sample_duration)

    rle_results = []
    for i in range(0, segments.shape[0]):
        _segment = segments[i]
        start = timedelta(seconds=round(_segment[-2].item(), 2))
        end = timedelta(seconds=round(_segment[-1].item(), 2))
        class_idx = int(_segment[2].item())
        if len(rle_results) == 0 or rle_results[-1]["class"] != idx2class_map[class_idx]:
            rle_results.append({"path": audio_filepath, "start": start, "end": end, "class": idx2class_map[class_idx]})
            continue
        rle_results[-1]["end"] = end

    df = pd.DataFrame(rle_results)
    if save_result:
        _path_split = audio_filepath.split((os.sep if os.sep in audio_filepath else "/"))
        filename = "".join(_path_split[-1].split(".")[:-1])
        output_dir = os.path.join(output_dir, "results")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        path = os.path.join(output_dir, f"{filename}_results.csv")
        df.to_csv(path, index=False)
        return df, path
    return df


async def async_evaluate_audio(semaphore: asyncio.Semaphore, **kwargs) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
    loop = asyncio.get_running_loop()
    async with semaphore:
        return await loop.run_in_executor(None, lambda : evaluate_audio(**kwargs))


async def async_evaluate_dir(
        model: AudioDetectionNetwork, 
        audio_dir: str, 
        output_dir: str,
        extensions: Iterable[str]=["wav", "mp3"], 
        num_concurrency: int=10, 
        **kwargs
    ) -> Union[List[pd.DataFrame], Tuple[List[pd.DataFrame], List[str]]]:
    audio_filepaths = []
    for ext in extensions:
        audio_filepaths += glob.glob(os.path.join(audio_dir, f"*.{ext}"))
    semaphore = asyncio.Semaphore(num_concurrency)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    tasks = [
        asyncio.ensure_future(
            async_evaluate_audio(semaphore, model=model, audio_filepath=path, output_dir=output_dir, **kwargs)
        ) for path in audio_filepaths
    ]

    dfs = []
    paths = []
    progress_bar = st.progress(0.0)
    progress = 0.0
    status = st.empty()
    status.text(f"Progress: {progress}%")
    for i, task in enumerate(asyncio.as_completed(tasks)):
        result = await task
        if isinstance(result, tuple):
            df, path = result
            dfs.append(df)
            paths.append(path)
        else:
            dfs.append(result)
        progress = (i + 1) / len(tasks)
        progress_bar.progress(progress)
        status.text(f"Progress: {progress*100 :.2f}%")
    status.text("100% Done!")
    if paths:
        return dfs, paths
    return dfs

def main(
        config: Dict[str, Any], 
        sample_file_or_dir: str,
        class_map_path: str,  
        model_path: str, 
        batch_size: Optional[int]=None,
        output_dir: str="segment_predictions",
        extensions: Iterable[str]=["wav", "mp3"],
        iou_threshold: float=0.1, 
        conf_threshold: float=0.2,
        num_concurrency: int=10,
        combine_results: bool=False
    ):
    if not model_path:
        block_config = "_".join(map(lambda i : str(i), config["block_layers"]))
        if config["backbone"] == "custom":
            model_path = os.path.join(
                config["train_config"]["model_path"], 
                "ivy-openbmat", 
                f"{config['backbone']}_{block_config}",
                f"{AudioDetectionNetwork.__name__}.pth.tar"
            )
        elif config['backbone'] == "resnet":
            model_path = os.path.join(
                config["train_config"]["model_path"], 
                "ivy-openbmat", 
                f"{config['backbone']}_{config['resnet_config']['block']}_{block_config}",
                f"{AudioDetectionNetwork.__name__}.pth.tar"
            )

    device = config["device"] if torch.cuda.is_available() else "cpu"
    if not os.path.isfile(class_map_path):
        raise FileNotFoundError(f"{class_map_path} file does not exist in {class_map_path}")
    idx2class_map = get_label_map(class_map_path)
    num_classes = len(idx2class_map)

    model = AudioDetectionNetwork(num_classes, config=config)
    load_model_weights(model, model_path=model_path, device=device)
    kwargs = dict(
        input_sample_rate=config["sample_rate"], 
        sample_duration=config["sample_duration"], 
        batch_size=batch_size or config["train_config"]["batch_size"], 
        idx2class_map=idx2class_map,
        device=device,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
    )
    if os.path.isfile(sample_file_or_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        _, path = evaluate_audio(model, sample_file_or_dir, output_dir, save_result=True, **kwargs)
        return path

    elif os.path.isdir(sample_file_or_dir):
        extensions = [ext.replace(".", "") for ext in extensions]
        kwargs["save_result"] = (not combine_results)
        results = asyncio.run(async_evaluate_dir(
            model, sample_file_or_dir, output_dir, extensions, num_concurrency, **kwargs)
        )
        if combine_results:
            results = pd.concat(results, axis=0)
            path = os.path.join(output_dir, "results.csv")
            results.to_csv(path, index=False)
            return path
        _, paths = results
        return paths
    else:
        raise OSError(f"file / directory {sample_file_or_dir} not found")

    return results