# Airplay Audio Activity Detection and Finger Printing

## Setup

1.  Clone the repository with `git clone https://github.com/ches-001/airplay-audio-fingerprinting`

2. Copy the folder containing the song files you wish to use to construct the vector DB into the cloned repository folder

3. Copy the folder containing the audio files you wish to run inference on into the cloned repository folder

4. Install all dependencies required by the repository in the requirements.txt file with the command `pip install -r requirements.txt`

5. Download / Retrieve the the relevant models and put then in their respective folders. The audio activity detection model goes into the `saved_model/resnet_BasicBlock_2_2_2_2` folder, and the fingerprinting model goes into the `baseline_model` folder. Both models have names "AudioDetectionNetwork.pth.tar" and "model.pt" respectively.

6. Run the streamlit application via the command `streamlit run interface.py`. This will spin up a streamlit service and generate 2 URLs, one for `Network URL: http://192.168.0.4:8501` and `Local URL: http://localhost:8501`. If you are on a local machine with a display, you can access the webpage via the localhost url. If you are on a remote server with restricted access, you can expose the local web server to the global internet via a tunneling service called [ngrok](https://ngrok.com/download) (follow the guides on this page to setup ngrok on you respective operating system)

7. Once you are done setting up ngrok, run it on a seperate terminal with the command `ngrok http <port>`, where port is the port that the streamlit application listens from (8501 by default) and copy the generated URL to any browser.


## Usage

There are 3 sections / pages on the displayed webpage

1. Create vectorDB

2. Generate Segments

3. Match Audio

In the first page, all you need to do is provide the folder path to the audio files / musics you need to use to generate the vector DB. You can also change the default parameters, or just leave them be. Note that this webapp does not support the uploading of files, all paths you provide must be valid paths on the machine where the streamlit application runs, preferrably all folders / directories should be accessed from within the cloned repository folder.

In the second page, provide the file or directory path to the sample audios you wish to generate the segments, you can leave the default parameters as they are unless you want to change them. When this part runs, you can copy the outputted path on the webpage and paste it in the next step.

Finally, provide the path you copied (it must be a CSV file path) to the next page and run it. It will generate the finger printing results in a CSV file.