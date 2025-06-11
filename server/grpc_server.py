#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, glob, copy, pickle, wave, subprocess, time, struct
from concurrent.futures import ThreadPoolExecutor

import grpc, numpy as np, cv2, torch, torch.nn.functional as F, av
from scipy.signal import resample_poly
from transformers import WhisperModel, WhisperFeatureExtractor

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT, "proto"))
from proto import lipsync_pb2, lipsync_pb2_grpc
from musetalk.utils.utils import load_all_model
from musetalk.utils.preprocessing import read_imgs
from musetalk.utils.blending import get_image_blending

# ─── parâmetros de áudio ────────────────────────────────────────────
OPUS_SR   = 48_000
MONO24_SR = 24_000
WHISPER_SR= 16_000

CHUNK_S   = 0.2                           # segundos por quadro (mude p/ 0.2, 0.04…)
CHUNK_SMP = int(MONO24_SR*CHUNK_S)        # 24 000
CHUNK_B   = CHUNK_SMP*2

CTX_SEC   = 3.84
CTX_SMP24 = int(MONO24_SR*CTX_SEC)        # 92 160
CTX_B24   = CTX_SMP24*2

SAVE_MP3  = False
if SAVE_MP3: os.makedirs("audios", exist_ok=True)

# ─── helpers básicos ───────────────────────────────────────────────
def stereo2mono(a): return a.mean(axis=0)
pcm16f32  = lambda b: np.frombuffer(b,"<i2").astype(np.float32)/32768
def pad_crop(t,n): cur=t.size(-1); return t if cur==n else (t[...,-n:] if cur>n else F.pad(t,(n-cur,0)))

# ------------------- Circular buffer -------------------------------
class PCMHistory:
    """Ring-buffer de 92 160 amostras (24 kHz)."""
    def __init__(self, max_bytes=CTX_B24):
        self.max = max_bytes
        self.buf = bytearray()
    def append(self, pcm24: bytes):
        self.buf += pcm24
        if len(self.buf) > self.max:
            del self.buf[:len(self.buf)-self.max]
    def get_window(self) -> bytes:
        if len(self.buf) < self.max:
            return b'\x00'*(self.max-len(self.buf)) + self.buf
        return bytes(self.buf)

# ------------------- RTP(Opus) → PCM24  ----------------------------
class RTPOpusDecoder:
    """PyAV com extradata OpusHead para payload RTP cru."""
    def __init__(self):
        self.ctx = av.CodecContext.create("opus","r")
        # 19-byte OpusHead
        self.ctx.extradata = (
            b'OpusHead' + b'\x01' + b'\x02' +
            struct.pack('<H',312) + struct.pack('<I',OPUS_SR) +
            struct.pack('<H',0) + b'\x00'
        )
    def __call__(self, pkt: bytes) -> bytes | None:
        if len(pkt)<12: return None
        frames = self.ctx.decode(av.Packet(pkt[12:]))
        if not frames: return None
        pcm = np.concatenate([f.to_ndarray() for f in frames], axis=1)  # (2,N) f32
        mono24 = stereo2mono(pcm)[::2]                                 # 48 k→24 k
        return (mono24*32767).astype(np.int16).tobytes()

# ------------------- Serviço gRPC ----------------------------------
class LipSyncServicer(lipsync_pb2_grpc.LipSyncServiceServicer):
    def __init__(self,cfg):
        self.cfg=cfg
        self.dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fp16=self.dev.type=="cuda"
        self.dtype=torch.float16 if self.fp16 else torch.float32

        self.decode = RTPOpusDecoder()
        self.chunk  = bytearray()
        self.hist   = PCMHistory()

        self.chunk_idx=0; self.fidx=0
        self._load_models(); self._load_avatar()
        self.t0=torch.tensor([0],device=self.dev)

    # ---- modelos/avatar -----------------------------------------------------
    def _load_models(self):
        self.vae,self.unet,self.pe = load_all_model(
            self.cfg.unet_model_path, self.cfg.vae_type, self.cfg.unet_config, device=self.dev)
        for m in (self.pe,self.vae.vae,self.unet.model):
            (m.half() if self.fp16 else m).to(self.dev)
        self.whisper = WhisperModel.from_pretrained(self.cfg.whisper_dir)\
                         .to(self.dev,dtype=self.dtype).eval()
        self.fe = WhisperFeatureExtractor.from_pretrained(self.cfg.whisper_dir,padding="do_not_pad")

    def _load_avatar(self):
        root=f"./results/{self.cfg.version}/avatars/{self.cfg.avatar_id}"
        self.lat=torch.load(f"{root}/latents.pt")
        self.coords      = pickle.load(open(f"{root}/coords.pkl","rb"))
        self.mask_coords = pickle.load(open(f"{root}/mask_coords.pkl","rb"))
        imgs=sorted(glob.glob(f"{root}/full_imgs/*"), key=lambda p:int(os.path.splitext(os.path.basename(p))[0]))
        masks=sorted(glob.glob(f"{root}/mask/*"),     key=lambda p:int(os.path.splitext(os.path.basename(p))[0]))
        self.frames=read_imgs(imgs); self.masks=read_imgs(masks)

    # ---- debug chunk → MP3 ---------------------------------------------------
    def _save_mp3(self, pcm24: bytes):
        if not SAVE_MP3: return
        tmp="audios/tmp.wav"; mp3=f"audios/chunk_{self.chunk_idx:06d}.mp3"
        with wave.open(tmp,"wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(MONO24_SR)
            w.writeframes(pcm24)
        subprocess.run(["ffmpeg","-y","-loglevel","quiet","-i",tmp,
                        "-codec:a","libmp3lame","-q:a","5", mp3], check=False)
        os.remove(tmp)

    # ---- gera JPEG ----------------------------------------------------------
    @torch.no_grad()
    def _make_frame(self)->bytes:
        # resample high-quality 24k → 16k
        f32_16 = resample_poly(pcm16f32(self.hist.get_window()), WHISPER_SR, MONO24_SR).astype(np.float32)
        mel = self.fe(f32_16, sampling_rate=WHISPER_SR, return_tensors="pt").input_features
        mel = pad_crop(mel, 384).to(self.dev, dtype=self.dtype)
        emb = self.pe(mel)

        lat=self.lat[self.fidx].to(self.dev,dtype=self.unet.model.dtype)
        pred=self.unet.model(lat,self.t0,encoder_hidden_states=emb).sample
        face=self.vae.decode_latents(pred.to(self.vae.vae.dtype))[0]

        x1,y1,x2,y2=self.coords[self.fidx]
        base=copy.deepcopy(self.frames[self.fidx])
        face_r=cv2.resize(face.astype(np.uint8),(x2-x1,y2-y1))
        blended=get_image_blending(base,face_r,(x1,y1,x2,y2),
                                   self.masks[self.fidx], self.mask_coords[self.fidx])
        ok,buf=cv2.imencode(".jpg",blended,[int(cv2.IMWRITE_JPEG_QUALITY),90])
        if not ok: raise RuntimeError("jpeg fail")
        self.fidx=(self.fidx+1)%len(self.lat)
        return buf.tobytes()

    # ---- stream RPC ---------------------------------------------------------
    def StreamAudioToVideo(self, iterator, ctx):
        try:
            for pkt in iterator:
                if not pkt.audio_data: continue
                pcm24=self.decode(pkt.audio_data)
                if pcm24 is None: continue

                self.chunk += pcm24

                while len(self.chunk) >= CHUNK_B:
                    piece = bytes(self.chunk[:CHUNK_B]); del self.chunk[:CHUNK_B]
                    self._save_mp3(piece); self.chunk_idx+=1
                    self.hist.append(piece)

                    jpeg=self._make_frame()
                    yield lipsync_pb2.VideoFrame(frame_data=jpeg,
                                                 timestamp=int(time.time()*1000))
            # flush restante
            if self.chunk:
                self.hist.append(self.chunk)
                yield lipsync_pb2.VideoFrame(frame_data=self._make_frame(),
                                             timestamp=int(time.time()*1000))
        except Exception as e:
            ctx.set_code(grpc.StatusCode.INTERNAL); ctx.set_details(str(e))

# ─── bootstrap ─────────────────────────────────────────────────────
def serve(cfg):
    srv=grpc.server(ThreadPoolExecutor(max_workers=10))
    lipsync_pb2_grpc.add_LipSyncServiceServicer_to_server(LipSyncServicer(cfg), srv)
    srv.add_insecure_port(f"[::]:{cfg.grpc_port}")
    srv.start(); print(f"Servidor ativo (janela circular 3.84 s, {CHUNK_S}s por frame)")
    srv.wait_for_termination()

if __name__=="__main__":
    import argparse, omegaconf
    p=argparse.ArgumentParser(); p.add_argument("--config", default="configs/grpc_server.yaml")
    serve(omegaconf.OmegaConf.load(p.parse_args().config))
