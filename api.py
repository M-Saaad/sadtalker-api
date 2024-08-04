import torch
import shutil
import os, sys, time
from glob import glob
from time import  strftime
from argparse import ArgumentParser

from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path

from fastapi import FastAPI, File, UploadFile
import uuid
import uvicorn

app = FastAPI()

def infer(
    # driven_audio: UploadFile = File(...),
    # source_image: UploadFile = File(...),
    driven_audio: str,
    source_image: str,
    ref_eyeblink: UploadFile = None,
    ref_pose: UploadFile = None,
    checkpoint_dir: str = './checkpoints',
    result_dir: str = './results',
    pose_style: int = 0,
    batch_size: int = 2,
    size: int = 512,
    expression_scale: float = 1.0,
    input_yaw: str = '',
    input_pitch: str = '',
    input_roll: str = '',
    enhancer: str = None,
    background_enhancer: str = None,
    cpu: bool = False,
    face3dvis: bool = False,
    still: bool = False,
    preprocess: str = 'crop',
    verbose: bool = False,
    old_version: bool = False,
    net_recon: str = 'resnet50',
    initial_path: str = None,
    use_last_fc: bool = False,
    bfm_folder: str = './checkpoints/BFM_Fitting/',
    bfm_model: str = 'BFM_model_front.mat',
    focal: float = 1015.0,
    center: float = 112.0,
    camera_d: float = 10.0,
    z_near: float = 5.0,
    z_far: float = 15.0,
    device: str = None
):
    
    print(source_image, driven_audio)
    
    #######################################################
    # audio_path = f"/tmp/{driven_audio.filename}"
    # image_path = f"/tmp/{source_image.filename}"

    # with open(audio_path, "wb") as audio_file:
    #     audio_file.write(driven_audio.file.read())

    # with open(image_path, "wb") as image_file:
    #     image_file.write(source_image.file.read())

    ref_eyeblink_path = None
    ref_pose_path = None
    if ref_eyeblink:
        ref_eyeblink_path = f"tmp/{ref_eyeblink.filename}"
        with open(ref_eyeblink_path, "wb") as ref_eyeblink_file:
            ref_eyeblink_file.write(ref_eyeblink.file.read())
    
    if ref_pose:
        ref_pose_path = f"tmp/{ref_pose.filename}"
        with open(ref_pose_path, "wb") as ref_pose_file:
            ref_pose_file.write(ref_pose.file.read())

    input_yaw_list = [int(yaw) for yaw in input_yaw.split(',')] if input_yaw else None
    input_pitch_list = [int(pitch) for pitch in input_pitch.split(',')] if input_pitch else None
    input_roll_list = [int(roll) for roll in input_roll.split(',')] if input_roll else None
    #######################################################

    #torch.backends.cudnn.enabled = False

    pic_path = source_image
    audio_path = driven_audio
    save_dir = os.path.join(result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = pose_style
    device = device
    batch_size = batch_size
    input_yaw_list = input_yaw_list
    input_pitch_list = input_pitch_list
    input_roll_list = input_roll_list
    ref_eyeblink = ref_eyeblink
    ref_pose = ref_pose

    if torch.cuda.is_available() and not device:
        device = "cuda"
    else:
        device = "cpu"

    print("DEVICE:", device)

    # current_root_path = os.path.split(sys.argv[0])[0]
    current_root_path = ""

    sadtalker_paths = init_path(checkpoint_dir, os.path.join(current_root_path, 'src/config'), size, old_version, preprocess)

    # print("######################", current_root_path)

    #init model
    preprocess_model = CropAndExtract(sadtalker_paths, device)

    audio_to_coeff = Audio2Coeff(sadtalker_paths,  device)
    
    animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    #crop image and extract 3dmm from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')
    first_coeff_path, crop_pic_path, crop_info =  preprocess_model.generate(pic_path, first_frame_dir, preprocess,\
                                                                             source_image_flag=True, pic_size=size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ =  preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_eyeblink_coeff_path=None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink: 
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ =  preprocess_model.generate(ref_pose, ref_pose_frame_dir, preprocess, source_image_flag=False)
    else:
        ref_pose_coeff_path=None

    #audio2ceoff
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=still)
    coeff_path = audio_to_coeff.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

    # 3dface render
    # if face3dvis:
    #     from src.face3d.visualize import gen_composed_video
    #     gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))
    
    #coeff2video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, 
                                batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                                expression_scale=expression_scale, still_mode=still, preprocess=preprocess, size=size)
    
    print(f"SAVE DIR: {save_dir}")
    print(f"PIC PATH: {pic_path}")

    result = animate_from_coeff.generate(data, save_dir, pic_path, crop_info, \
                                enhancer=enhancer, background_enhancer=background_enhancer, preprocess=preprocess, img_size=size)
    
    shutil.move(result, save_dir+'.mp4')
    print(result)
    print('The generated video is named:', save_dir+'.mp4')

    if not verbose:
        shutil.rmtree(save_dir)

    return save_dir