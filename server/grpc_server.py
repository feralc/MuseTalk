#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, glob, copy, pickle, wave, subprocess, struct
from concurrent.futures import ThreadPoolExecutor

import grpc, numpy as np, cv2, torch, torch.nn.functional as F, av
from transformers import WhisperModel, WhisperFeatureExtractor

# ─── paths ────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "proto"))
from proto import lipsync_pb2, lipsync_pb2_grpc
from musetalk.utils.utils import load_all_model
from musetalk.utils.preprocessing import read_imgs
from musetalk.utils.blending import get_image_blending

# ─── audio constants ─────────────────────────────────────────────
OPUS_SR   = 48_000
MONO24_SR = 24_000
WHISPER_SR= 16_000

CHUNK_S   = 1                              # process every 1 second
CHUNK_SMP = MONO24_SR * CHUNK_S            # 24 000
CHUNK_B   = CHUNK_SMP * 2                  # bytes

CTX_SEC   = 3.84
CTX_SMP   = int(MONO24_SR * CTX_SEC)       # 92 160
CTX_B     = CTX_SMP * 2

SAVE_MP3  = False
if SAVE_MP3:
    os.makedirs("audios", exist_ok=True)

# ─── helpers — quick DSP —─────────────────────────────────────────
def stereo_to_mono(arr: np.ndarray) -> np.ndarray: return arr.mean(axis=0)
def decimate(x: np.ndarray, k: int) -> np.ndarray:  return x[::k]
pcm16_to_f32 = lambda b: np.frombuffer(b, "<i2").astype(np.float32) / 32768
def pad_left(data: bytes, total: int) -> bytes:
    return b'\x00'*(total-len(data)) + data if len(data)<total else data[-total:]

def pad_crop(t: torch.Tensor, n: int) -> torch.Tensor:
    cur = t.size(-1)
    return t if cur==n else (t[...,-n:] if cur>n else F.pad(t,(n-cur,0)))

# ─── RTP/Opus → mono PCM 24 k decoder (PyAV) ──────────────────────
class RTPToPCM24:
    def __init__(self):
        self.codec = av.CodecContext.create("opus", "r")
    def __call__(self, rtp: bytes) -> bytes | None:
        if len(rtp) < 12:
            return None
        frames = self.codec.decode(av.packet.Packet(rtp[12:]))
        if not frames:
            return None
        pcm = np.concatenate([f.to_ndarray() for f in frames], axis=1)  # (2,N) float32
        mono48 = stereo_to_mono(pcm)
        mono24 = decimate(mono48, 2)
        return (mono24 * 32767).astype(np.int16).tobytes()

# ─── gRPC service ─────────────────────────────────────────────────
class LipSyncServicer(lipsync_pb2_grpc.LipSyncServiceServicer):
    def __init__(self, cfg):
        self.cfg  = cfg
        self.dev  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fp16 = self.dev.type == "cuda"
        self.dtype= torch.float16 if self.fp16 else torch.float32

        self.decode = RTPToPCM24()
        self.chunk_buf = bytearray()   # 1-second buffer
        self.ctx_buf   = bytearray()   # last 3.84 s

        self.chunk_idx = 0
        self.frame_idx = 0

        self._load_models(); self._load_avatar()
        self.t0 = torch.tensor([0], device=self.dev)

    def _load_models(self):
        self.vae, self.unet, self.pe = load_all_model(
            self.cfg.unet_model_path, self.cfg.vae_type, self.cfg.unet_config, device=self.dev)
        for m in (self.pe, self.vae.vae, self.unet.model):
            (m.half_() if self.fp16 else m).to(self.dev)
        self.whisper = WhisperModel.from_pretrained(self.cfg.whisper_dir)\
                         .to(self.dev, dtype=self.dtype).eval()
        self.fe = WhisperFeatureExtractor.from_pretrained(
            self.cfg.whisper_dir, padding="do_not_pad")

    def _load_avatar(self):
        root = f"./results/{self.cfg.version}/avatars/{self.cfg.avatar_id}"
        self.lat = torch.load(f"{root}/latents.pt")
        self.coords      = pickle.load(open(f"{root}/coords.pkl","rb"))
        self.mask_coords = pickle.load(open(f"{root}/mask_coords.pkl","rb"))
        imgs  = sorted(glob.glob(f"{root}/full_imgs/*"),
                       key=lambda p:int(os.path.splitext(os.path.basename(p))[0]))
        masks = sorted(glob.glob(f"{root}/mask/*"),
                       key=lambda p:int(os.path.splitext(os.path.basename(p))[0]))
        self.frames = read_imgs(imgs); self.masks = read_imgs(masks)

    # optional debug
    def _save_mp3(self, pcm24: bytes):
        if not SAVE_MP3: return
        tmp, mp3 = "tmp.wav", f"audios/chunk_{self.chunk_idx:06d}.mp3"
        with wave.open(tmp,"wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(MONO24_SR)
            w.writeframes(pcm24)
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",tmp,
                        "-codec:a","libmp3lame","-qscale:a","4",
                        "-ar",str(MONO24_SR), mp3], check=False)
        os.remove(tmp)

    @torch.no_grad()
    def _make_frame(self) -> bytes:
        ctx_f32 = decimate(pcm16_to_f32(self.ctx_buf), MONO24_SR // WHISPER_SR)
        mel = self.fe(ctx_f32, sampling_rate=WHISPER_SR, return_tensors="pt").input_features
        mel = pad_crop(mel, 384).to(self.dev, dtype=self.dtype)
        emb = self.pe(mel)

        lat = self.lat[self.frame_idx].to(self.dev, dtype=self.unet.model.dtype)
        pred = self.unet.model(lat, self.t0, encoder_hidden_states=emb).sample
        face = self.vae.decode_latents(pred.to(self.vae.vae.dtype))[0]

        x1,y1,x2,y2 = self.coords[self.frame_idx]
        base  = copy.deepcopy(self.frames[self.frame_idx])
        face_r= cv2.resize(face.astype(np.uint8),(x2-x1,y2-y1))
        blended= get_image_blending(base,face_r,(x1,y1,x2,y2),
                                    self.masks[self.frame_idx], self.mask_coords[self.frame_idx])
        ok, buf= cv2.imencode(".jpg", blended,[int(cv2.IMWRITE_JPEG_QUALITY),90])
        if not ok: raise RuntimeError("JPEG encode failed")
        self.frame_idx = (self.frame_idx + 1) % len(self.lat)
        return buf.tobytes()

    # ─── duplex streaming RPC ───────────────────────────────────────
    def StreamAudioToVideo(self, iterator, ctx):
        try:
            for rtp in iterator:
                if not rtp.audio_data: continue
                pcm24 = self.decode(rtp.audio_data)
                if pcm24 is None: continue

                self.chunk_buf += pcm24

                while len(self.chunk_buf) >= CHUNK_B:
                    chunk = bytes(self.chunk_buf[:CHUNK_B])
                    del self.chunk_buf[:CHUNK_B]

                    self._save_mp3(chunk)
                    self.chunk_idx += 1

                    # update 3.84-s context
                    self.ctx_buf = pad_left(self.ctx_buf + chunk, CTX_B)

                    jpeg = self._make_frame()
                    yield lipsync_pb2.VideoFrame(frame_data=jpeg,
                                                 timestamp=int(time.time()*1000))
            # flush leftover (<1 s) at stream end
            if self.chunk_buf:
                chunk = bytes(self.chunk_buf)
                self._save_mp3(chunk)
                self.ctx_buf = pad_left(self.ctx_buf + chunk, CTX_B)
                yield lipsync_pb2.VideoFrame(frame_data=self._make_frame(),
                                             timestamp=int(time.time()*1000))
        except Exception as e:
            ctx.set_code(grpc.StatusCode.INTERNAL); ctx.set_details(str(e))

# ─── bootstrap ─────────────────────────────────────────────────────
def serve(cfg):
    server = grpc.server(ThreadPoolExecutor(max_workers=10))
    lipsync_pb2_grpc.add_LipSyncServiceServicer_to_server(LipSyncServicer(cfg), server)
    server.add_insecure_port(f"[::]:{cfg.grpc_port}")
    server.start(); print(f"Lip-sync server ativo na porta {cfg.grpc_port}")
    server.wait_for_termination()

if __name__ == "__main__":
    import argparse, omegaconf
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/grpc_server.yaml")
    serve(omegaconf.OmegaConf.load(p.parse_args().config))
