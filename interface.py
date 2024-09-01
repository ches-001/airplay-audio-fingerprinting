import os
import torch
import logging
import warnings
import torch.multiprocessing as mp
from inference import load_config, main as seg_gen_main
from builder import build_vector_db
from matcher import match_samples
from modules._architecture import AudioDetectionNetwork
import streamlit as st


warnings.filterwarnings(action="ignore")
logger = logging.getLogger(__name__)


if __name__ == "__main__": 
    MANAGE_VECTOR_DB_PAGE = "Create / Update VectorDB"
    MATCH_PAGE = "Match Audios"
    INFERENCE_PAGE = "File Inference"
    SUPPORTED_EXTENSIONS = ["mp3", "wav"]

    page = st.sidebar.radio(
        "Select a page", [MANAGE_VECTOR_DB_PAGE, INFERENCE_PAGE, MATCH_PAGE]
    )

    config = load_config()
    batch_size = config["train_config"]["batch_size"]
    class_map_path = os.path.join(config["train_config"]["class_map_path"], "class_map.json")
    block_config = "_".join(map(lambda i : str(i), config["block_layers"]))
    if config["backbone"] == "custom":
        model_path = os.path.join(
            config["train_config"]["model_path"], 
            f"{config['backbone']}_{block_config}",
            f"{AudioDetectionNetwork.__name__}.pth.tar"
        )
    elif config['backbone'] == "resnet":
        model_path = os.path.join(
            config["train_config"]["model_path"], 
            f"{config['backbone']}_{config['resnet_config']['block']}_{block_config}",
            f"{AudioDetectionNetwork.__name__}.pth.tar"
        )
    else:
        logger.error("Invalid backbone")
    device = config["train_config"]["device"] if torch.cuda.is_available() else "cpu"
    output_dir = "final_results"
    num_concurrency = 10
    iou_threshold = 0.1
    conf_threshold = 0.2
    db_dir = "vector_db"
    db_config_path = "pfann_configs/default.json"

    if page == MANAGE_VECTOR_DB_PAGE:
        st.title("Create VectorDB")
        audios_dir_for_db = st.text_input("Directory of files to create vectorDB", value="")
        db_dir = st.text_input("VectorDB directory", value=db_dir)
        db_config_path = st.text_input("Config file", value=db_config_path)
        if st.button("Create VectorDB"):
            build_vector_db(audios_dir_for_db, db_dir, db_config_path, extensions=SUPPORTED_EXTENSIONS)

    elif page == INFERENCE_PAGE:
        st.title("Generate Segments Meta")
        sample_file_or_dir = st.text_input("File/Directory path", value="")
        class_map_path = st.text_input("Label map path", value=class_map_path)
        batch_size = int(st.text_input("Batch segments", value=batch_size))
        output_dir = st.text_input("Output directory", value=os.path.join(output_dir, "seg_results"))
        num_concurrency = int(st.text_input("Concurrency", value=num_concurrency))
        iou_threshold = float(st.text_input("IoU Threshold", value=iou_threshold))
        conf_threshold = float(st.text_input("Confidence Threshold", value=conf_threshold))
        combine_results = st.radio("Combine all results (applies to directories)", ("Yes", "No"))
        
        if st.button("Generate Segments"):
            path_or_paths = seg_gen_main(
                config, 
                sample_file_or_dir=sample_file_or_dir,
                class_map_path=class_map_path, 
                batch_size=batch_size,
                model_path=model_path, 
                output_dir=output_dir,
                iou_threshold=iou_threshold,
                conf_threshold=conf_threshold,
                num_concurrency=num_concurrency,
                combine_results=(combine_results == "Yes"),
                extensions=SUPPORTED_EXTENSIONS
            )     
            st.header("Result Path / Directory")
            if isinstance(path_or_paths, str):
                st.code(path_or_paths)
            else:
                st.code("".join(path_or_paths[0].split(os.sep)[:-1]))

    elif page == MATCH_PAGE:
        st.title("Match Generated Segments")
        segments_file = st.text_input("Segemnts CSV File", value="")
        db_dir = st.text_input("VectorDB directory", value=db_dir)

        if st.button("Match Segments"):
            result_file = segments_file.replace("seg_results", "pfnn_results")
            match_samples(segments_file, db_dir, result_file)
            st.code(result_file)