import os, sys
import time, re, json, shutil
import requests, subprocess
import yaml
from yaml.loader import SafeLoader
import io, base64
from PIL import Image, PngImagePlugin
import argparse

#import gradio as gr
#import spaces

from model import Model
from settings import CACHE_EXAMPLES, MAX_SEED
from utils import randomize_seed_fn
model = Model()

os.environ['CURL_CA_BUNDLE'] = ''
parser = argparse.ArgumentParser()
parser.add_argument("-mr", "--model_req", 
                    help="DeSOTA Request as yaml file path",
                    type=str)
parser.add_argument("-mru", "--model_res_url",
                    help="DeSOTA API Result URL. Recognize path instead of url for desota tests", # check how is atribuited the test_mode variable in main function
                    type=str)
parser.add_argument("-deb", "--debug",
                    help="DeSOTAdebug", # check how is atribuited the test_mode variable in main function
                    type=int, default=0)
parser.add_argument("-p", "--dprompt",
                    help="fast prompt for debug", # check how is atribuited the test_mode variable in main function
                    type=str, default='super mario hat')

DEBUG = False

# DeSOTA Funcs [START]
#   > Import DeSOTA Scripts
from desota import detools
#   > Grab DeSOTA Paths
USER_SYS = detools.get_platform()
APP_PATH = os.path.dirname(os.path.realpath(__file__))
#   > USER_PATH
if USER_SYS == "win":
    path_split = str(APP_PATH).split("\\")
    desota_idx = [ps.lower() for ps in path_split].index("desota")
    USER=path_split[desota_idx-1]
    USER_PATH = "\\".join(path_split[:desota_idx])
elif USER_SYS == "lin":
    path_split = str(APP_PATH).split("/")
    desota_idx = [ps.lower() for ps in path_split].index("desota")
    USER=path_split[desota_idx-1]
    USER_PATH = "/".join(path_split[:desota_idx])
DESOTA_ROOT_PATH = os.path.join(USER_PATH, "Desota")
# DeSOTA Funcs [END]


def main(args):
    '''
    return codes:
    0 = SUCESS
    1 = INPUT ERROR
    2 = OUTPUT ERROR
    3 = API RESPONSE ERROR
    9 = REINSTALL MODEL (critical fail)
    '''
    # Time when grabed
    _report_start_time = time.time()
    start_time = int(_report_start_time)

    #---INPUT---# TODO (PRO ARGS)
    #---INPUT---#
    req_text = ''
    if args.debug == 0:
    # DeSOTA Model Request
        model_request_dict = detools.get_model_req(args.model_req)

        # API Response URL
        result_id = args.model_res_url
        
        # TARGET File Path
        dir_path = os.path.dirname(os.path.realpath(__file__))
        out_filepath = os.path.join(dir_path, f"txt-to-img-{start_time}.png")
        out_urls = detools.get_url_from_str(result_id)
        if len(out_urls)==0:
            test_mode = True
            report_path = result_id
        else:
            test_mode = False
            send_task_url = out_urls[0]

        # Get html file
        req_text = detools.get_request_text(model_request_dict)

        if isinstance(req_text, list):
            req_text = ' '.join(req_text)

    
    if req_text or args.debug == 1:
        if args.debug == 1:
            req_text = args.dprompt
            model_request_dict = {
                'input_args':{}
                }
        default_config = {
        #"prompt": str(args.text_prompt),
        "guidance_scale":15.0,
        "seed":420,
        "randomize_seed":True,
        "num_inference_steps":40
        }
        targs = {}
        if 'model_args' in model_request_dict['input_args']:
            targs = model_request_dict['input_args']['model_args']

        if 'model_args' in model_request_dict['input_args']:
            targs = model_request_dict['input_args']['model_args']
        if 'prompt' in targs:
            if targs['prompt'] == '$initial-prompt$':
                targs['prompt'] = req_text
        else:
            targs['prompt'] = req_text
        payload = {**default_config, **targs}
        outfile = model.run_text(payload['prompt'], payload['seed'], payload['guidance_scale'], payload['num_inference_steps'])
        
    if not outfile:
        print(f"[ ERROR ] -> DeSOTA SD.Next API Output ERROR: No results can be parsed for this request")
        exit(2)
        
    #print(f"[ INFO ] -> Response:\n{json.dumps(r, indent=2)}")
    
    if test_mode:
        if not report_path.endswith(".json"):
            report_path += ".json"
        with open(report_path, "w") as rw:
            json.dump(
                {
                    "Model Result Path": outfile,
                    "Processing Time": time.time() - _report_start_time
                },
                rw,
                indent=2
            )
        detools.user_chown(report_path)
        #detools.user_chown(outfile)
        print(f"Path to report:\n\t{report_path}")
    else:
        files = []
        with open(outfile, 'rb') as fr:
            files.append(('upload[]', fr))
            # DeSOTA API Response Post
            send_task = requests.post(url = send_task_url, files=files)
            print(f"[ INFO ] -> DeSOTA API Upload Res:\n{json.dumps(send_task.json(), indent=2)}")
        # Delete temporary file
        os.remove(outfile)

        if send_task.status_code != 200:
            print(f"[ ERROR ] -> DeSOTA SD.Next API Post Failed (Info):\nfiles: {files}\nResponse Code: {send_task.status_code}")
            exit(3)
    
    print("TASK OK!")
    exit(0)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)