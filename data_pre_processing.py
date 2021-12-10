import cv2
import torch
import numpy as np
import os
import shutil
from tqdm import tqdm

def verify(path, n_frames):
    num_files = len(os.listdir(path))
    if num_files != n_frames:
        print(path, num_files)

def convert_video_to_frames(dir, video_name, n_frames=16):
    cap = cv2.VideoCapture(os.path.join(dir, video_name))   #   create video object
    v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) )              #   get number of frames
    frame_list = np.linspace(0, v_len-1, n_frames, dtype=np.int16)    #   generate frame indices list

    dir_name = video_name.split('.')[0]  #   extract file name
    save_path = os.path.join(dir, dir_name)
    # if os.path.exists(save_path):
    #     print(save_path)
    #     shutil.rmtree(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for idx in frame_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        frame_path = '{}/{}.jpg'.format(save_path, idx)
        if ret:
            cv2.imwrite(frame_path, frame)     # save frame as JPG file
        else:
            tmp_id = idx - n_frames/2
            cap.set(cv2.CAP_PROP_POS_FRAMES, tmp_id)
            ret, frame = cap.read()
            frame_path = '{}/{}.jpg'.format(save_path, tmp_id)
            if ret:
                cv2.imwrite(frame_path, frame)     # save frame as JPG file
            else:
                print(frame_path, idx, tmp_id, v_len, n_frames, frame_list)
    
    verify(save_path, n_frames)

    cap.release()

    


if __name__ == "__main__":
    train_path = 'data/train/'
    valid_path = 'data/valid/'

    for phase in [train_path, valid_path]:
        classes = os.listdir(phase)
        for c in classes:
            print(c)
            folder = os.path.join(phase, c)
            for f in tqdm(os.listdir(folder), total=len(os.listdir(folder))):
                convert_video_to_frames(folder, f)