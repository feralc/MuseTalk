import grpc
import concurrent.futures
import time
import io
import numpy as np
import cv2
import torch
import threading
import queue
import pickle
import glob
import copy
from concurrent.futures import ThreadPoolExecutor

# Import generated gRPC code
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proto import lipsync_pb2
from proto import lipsync_pb2_grpc

# Import MuseTalk components
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs
from musetalk.utils.blending import get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel

class LipSyncServicer(lipsync_pb2_grpc.LipSyncServiceServicer):
    def __init__(self, config):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.initialize_models()
        self.frame_queue = queue.Queue(maxsize=100)
        self.processing_thread = None
        self.is_processing = False

    def initialize_models(self):
        # Load model weights
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=self.config.unet_model_path,
            vae_type=self.config.vae_type,
            unet_config=self.config.unet_config,
            device=self.device
        )
        self.timesteps = torch.tensor([0], device=self.device)

        # Convert models to half precision
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)

        # Initialize audio processor and Whisper model
        self.audio_processor = AudioProcessor(feature_extractor_path=self.config.whisper_dir)
        self.weight_dtype = self.unet.model.dtype
        self.whisper = WhisperModel.from_pretrained(self.config.whisper_dir)
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        # Initialize face parser
        self.fp = FaceParsing(
            left_cheek_width=self.config.left_cheek_width,
            right_cheek_width=self.config.right_cheek_width
        )

        # Initialize avatar
        self.initialize_avatar()

    def initialize_avatar(self):
        # Load avatar data
        self.avatar_path = f"./results/{self.config.version}/avatars/{self.config.avatar_id}"
        self.input_latent_list_cycle = torch.load(f"{self.avatar_path}/latents.pt")
        with open(f"{self.avatar_path}/coords.pkl", 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        
        input_img_list = glob.glob(os.path.join(f"{self.avatar_path}/full_imgs", '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)
        
        with open(f"{self.avatar_path}/mask_coords.pkl", 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        
        input_mask_list = glob.glob(os.path.join(f"{self.avatar_path}/mask", '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(input_mask_list)

    def process_audio_chunk(self, audio_chunk):
        # Convert audio chunk to numpy array
        audio_data = np.frombuffer(audio_chunk.audio_data, dtype=np.float32)
        
        # Process audio with Whisper
        whisper_input_features = self.audio_processor.process_audio_chunk(
            audio_data,
            sample_rate=audio_chunk.sample_rate,
            weight_dtype=self.weight_dtype
        )
        
        # Generate video frames
        audio_feature = self.pe(whisper_input_features.to(self.device))
        latent = self.input_latent_list_cycle[0].to(device=self.device, dtype=self.unet.model.dtype)
        
        pred_latents = self.unet.model(
            latent,
            self.timesteps,
            encoder_hidden_states=audio_feature
        ).sample
        
        pred_latents = pred_latents.to(device=self.device, dtype=self.vae.vae.dtype)
        recon = self.vae.decode_latents(pred_latents)
        
        # Process and blend the generated frame
        frame_idx = 0
        bbox = self.coord_list_cycle[frame_idx]
        ori_frame = copy.deepcopy(self.frame_list_cycle[frame_idx])
        x1, y1, x2, y2 = bbox
        
        res_frame = cv2.resize(recon[0].astype(np.uint8), (x2 - x1, y2 - y1))
        mask = self.mask_list_cycle[frame_idx]
        mask_crop_box = self.mask_coords_list_cycle[frame_idx]
        
        combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
        
        # Encode frame as JPEG
        _, jpeg_frame = cv2.imencode('.jpg', combine_frame)
        return jpeg_frame.tobytes()

    def StreamAudioToVideo(self, request_iterator, context):
        try:
            for audio_chunk in request_iterator:
                # Process audio chunk and generate video frame
                frame_data = self.process_audio_chunk(audio_chunk)
                
                # Create and yield video frame response
                response = lipsync_pb2.VideoFrame(
                    frame_data=frame_data,
                    timestamp=int(time.time() * 1000)
                )
                yield response
                
        except Exception as e:
            print(f"Error in StreamAudioToVideo: {str(e)}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))

def serve(config):
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    lipsync_pb2_grpc.add_LipSyncServiceServicer_to_server(
        LipSyncServicer(config), server
    )
    server.add_insecure_port(f'[::]:{config.grpc_port}')
    server.start()
    print(f"Server started on port {config.grpc_port}")
    server.wait_for_termination()

if __name__ == '__main__':
    import argparse
    from omegaconf import OmegaConf
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpc_server.yaml")
    args = parser.parse_args()
    
    config = OmegaConf.load(args.config)
    serve(config) 