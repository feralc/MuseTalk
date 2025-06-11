import math
import os

import librosa
import numpy as np
import torch
from einops import rearrange
from transformers import AutoFeatureExtractor


class AudioProcessor:
    def __init__(self, feature_extractor_path="openai/whisper-tiny/"):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_path)

    def get_audio_feature(self, wav_path, start_index=0, weight_dtype=None):
        if not os.path.exists(wav_path):
            return None
        librosa_output, sampling_rate = librosa.load(wav_path, sr=16000)
        assert sampling_rate == 16000
        # Split audio into 30s segments
        segment_length = 30 * sampling_rate
        segments = [librosa_output[i:i + segment_length] for i in range(0, len(librosa_output), segment_length)]

        features = []
        for segment in segments:
            audio_feature = self.feature_extractor(
                segment,
                return_tensors="pt",
                sampling_rate=sampling_rate
            ).input_features
            if weight_dtype is not None:
                audio_feature = audio_feature.to(dtype=weight_dtype)
            features.append(audio_feature)

        return features, len(librosa_output)

    def get_whisper_chunk(
        self,
        whisper_input_features,
        device,
        weight_dtype,
        whisper,
        librosa_length,
        fps=25,
        audio_padding_length_left=2,
        audio_padding_length_right=2,
    ):
        audio_feature_length_per_frame = 2 * (audio_padding_length_left + audio_padding_length_right + 1)
        whisper_feature = []
        # Process multiple 30s mel input features
        for input_feature in whisper_input_features:
            input_feature = input_feature.to(device).to(weight_dtype)
            audio_feats = whisper.encoder(input_feature, output_hidden_states=True).hidden_states
            audio_feats = torch.stack(audio_feats, dim=2)
            whisper_feature.append(audio_feats)

        whisper_feature = torch.cat(whisper_feature, dim=1)
        # Trim the last segment to remove padding
        sr = 16000
        audio_fps = 50
        fps = int(fps)
        whisper_idx_multiplier = audio_fps / fps
        num_frames = math.floor((librosa_length / sr) * fps)
        actual_length = math.floor((librosa_length / sr) * audio_fps)
        whisper_feature = whisper_feature[:,:actual_length,...]

        # Calculate padding amount
        padding_nums = math.ceil(whisper_idx_multiplier)
        # Add padding at start and end
        whisper_feature = torch.cat([
            torch.zeros_like(whisper_feature[:, :padding_nums * audio_padding_length_left]),
            whisper_feature,
            # Add extra padding to prevent out of bounds
            torch.zeros_like(whisper_feature[:, :padding_nums * 3 * audio_padding_length_right])
        ], 1)

        audio_prompts = []
        for frame_index in range(num_frames):
            try:
                audio_index = math.floor(frame_index * whisper_idx_multiplier)
                audio_clip = whisper_feature[:, audio_index: audio_index + audio_feature_length_per_frame]
                assert audio_clip.shape[1] == audio_feature_length_per_frame
                audio_prompts.append(audio_clip)
            except Exception as e:
                print(f"Error occurred: {e}")
                print(f"whisper_feature.shape: {whisper_feature.shape}")
                print(f"audio_clip.shape: {audio_clip.shape}")
                print(f"num frames: {num_frames}, fps: {fps}, whisper_idx_multiplier: {whisper_idx_multiplier}")
                print(f"frame_index: {frame_index}, audio_index: {audio_index}-{audio_index + audio_feature_length_per_frame}")
                exit()

        audio_prompts = torch.cat(audio_prompts, dim=0)  # T, 10, 5, 384
        audio_prompts = rearrange(audio_prompts, 'b c h w -> b (c h) w')
        return audio_prompts

    def build_audio_prompt_from_samples(
        self,
        audio_samples,
        device,
        weight_dtype,
        whisper: "WhisperModel",
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2,
    ):
        """Generate a single audio prompt (shape = [1, 50, 384]) from raw 16-kHz samples.

        This utility mirrors the logic inside ``get_whisper_chunk`` that is used for
        the offline pipeline, but is adapted for realtime streaming. Given the
        most recent slice of audio samples (e.g. the 3.84-second circular
        buffer in the gRPC server) it returns the 10-frame × 5-layer whisper
        embeddings that correspond to the *latest* video frame.

        Args:
            audio_samples (Union[np.ndarray, torch.Tensor]): 1-D float32 array in
                the range ‑1…1 sampled at 16 kHz.
            device (torch.device): Target device for whisper/inference.
            weight_dtype (torch.dtype): torch.float16 or torch.float32 matching
                the rest of the network.
            whisper (transformers.WhisperModel): Loaded whisper encoder.
            audio_padding_length_left (int): Temporal padding (frames) to the
                left of the current frame.
            audio_padding_length_right (int): Temporal padding (frames) to the
                right of the current frame.

        Returns:
            torch.Tensor: shape (1, 50, 384) ready to be fed into the prompt
                encoder ``pe``.
        """
        import torch
        from einops import rearrange

        # Ensure numpy → torch tensor
        if not torch.is_tensor(audio_samples):
            audio_samples = torch.from_numpy(audio_samples)
        audio_samples = audio_samples.to(device)

        # Whisper expects float32
        if audio_samples.dtype != torch.float32:
            audio_samples = audio_samples.float()

        # Convert raw audio to Whisper log-mel features using the same
        # AutoFeatureExtractor utilised in the offline pipeline.
        with torch.no_grad():
            input_feature = self.feature_extractor(
                audio_samples.cpu().numpy(),  # extractor works on numpy
                return_tensors="pt",
                sampling_rate=16000,
            ).input_features.to(device)
            input_feature = input_feature.to(weight_dtype)

            # Run Whisper encoder and gather all hidden states
            hidden_states = whisper.encoder(
                input_feature, output_hidden_states=True
            ).hidden_states  # tuple(L) of (1, T, 384)

            # Stack into (1, T, L, 384)
            hidden = torch.stack(hidden_states, dim=2)

            # Total temporal frames in hidden representation
            total_t = hidden.shape[1]
            audio_feature_length_per_frame = 2 * (
                audio_padding_length_left + audio_padding_length_right + 1
            )  # normally 10

            # If we do not have enough context yet, pad with zeros on the left
            if total_t < audio_feature_length_per_frame:
                pad_t = audio_feature_length_per_frame - total_t
                pad = torch.zeros(
                    hidden.shape[0],
                    pad_t,
                    hidden.shape[2],
                    hidden.shape[3],
                    dtype=hidden.dtype,
                    device=hidden.device,
                )
                hidden = torch.cat([pad, hidden], dim=1)
                total_t = hidden.shape[1]

            # Take the latest window (right-aligned)
            audio_clip = hidden[:, -audio_feature_length_per_frame :]
            # Rearrange to (1, 50, 384) – 10 temporal × 5 layer concat on channel dim
            audio_prompt = rearrange(audio_clip, "b c h w -> b (c h) w")
            return audio_prompt

    def build_audio_prompt_for_frame(
        self,
        audio_samples,
        frame_index: int,
        fps: int,
        device,
        weight_dtype,
        whisper: "WhisperModel",
        audio_padding_length_left: int = 2,
        audio_padding_length_right: int = 2,
    ):
        """Return whisper prompt aligned to *frame_index* for a video running at *fps*.

        This mirrors ``get_whisper_chunk`` from the offline pipeline but avoids
        the overhead of processing the entire audio file for every frame – it
        only keeps a short buffer (e.g. 10 s) that is already in memory.

        Args:
            audio_samples (Union[np.ndarray, torch.Tensor]): Mono 16 kHz float32
                waveform (range ‑1..1).
            frame_index (int): Index of the video frame we want to generate
                (0-based since the beginning of the current buffer).
            fps (int): Video frames-per-second (usually 25).
            device (torch.device): Computation device.
            weight_dtype (torch.dtype): dtype for Whisper and downstream.
            whisper (WhisperModel): Pre-loaded Whisper model (encoder is used).
            audio_padding_length_left/right (int): Padding lengths like the
                offline implementation.

        Returns:
            torch.Tensor of shape (1, 50, 384) ready for the prompt encoder.
        """
        import torch
        from einops import rearrange
        import math

        if not torch.is_tensor(audio_samples):
            audio_samples = torch.from_numpy(audio_samples)
        audio_samples = audio_samples.to(device, dtype=torch.float32)

        with torch.no_grad():
            mel_input = self.feature_extractor(
                audio_samples.cpu().numpy(),
                return_tensors="pt",
                sampling_rate=16000,
            ).input_features.to(device).to(weight_dtype)

            hidden_states = whisper.encoder(mel_input, output_hidden_states=True).hidden_states
            hidden = torch.stack(hidden_states, dim=2)  # (1, T, 10, 5, 384)

            sr = 16000
            audio_fps = 50
            whisper_idx_multiplier = audio_fps / fps  # usually 2

            # Determine audio_feature_length_per_frame (always 10)
            audio_feature_length_per_frame = 2 * (
                audio_padding_length_left + audio_padding_length_right + 1
            )

            # Trim hidden to actual length (safety)
            actual_length = hidden.shape[1]

            # Padding (same as offline)
            padding_nums = math.ceil(whisper_idx_multiplier)
            hidden = torch.cat([
                torch.zeros_like(hidden[:, : padding_nums * audio_padding_length_left]),
                hidden,
                torch.zeros_like(hidden[:, : padding_nums * 3 * audio_padding_length_right]),
            ], dim=1)

            # Calculate where this frame should read in audio timeline
            audio_index = math.floor(frame_index * whisper_idx_multiplier)
            audio_clip = hidden[:, audio_index : audio_index + audio_feature_length_per_frame]
            if audio_clip.shape[1] != audio_feature_length_per_frame:
                # Not enough future context yet – pad with zeros on the right
                pad_t = audio_feature_length_per_frame - audio_clip.shape[1]
                audio_clip = torch.cat([
                    audio_clip,
                    torch.zeros(
                        audio_clip.shape[0],
                        pad_t,
                        audio_clip.shape[2],
                        audio_clip.shape[3],
                        dtype=audio_clip.dtype,
                        device=audio_clip.device,
                    ),
                ], dim=1)

            prompt = rearrange(audio_clip, "b c h w -> b (c h) w")
            return prompt

if __name__ == "__main__":
    audio_processor = AudioProcessor()
    wav_path = "./2.wav"
    audio_feature, librosa_feature_length = audio_processor.get_audio_feature(wav_path)
    print("Audio Feature shape:", audio_feature.shape)
    print("librosa_feature_length:", librosa_feature_length)

