import gc
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from pydub import AudioSegment

from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data
from src.test_audio2coeff import Audio2Coeff
from src.utils.init_path import init_path
from src.utils.preprocess import CropAndExtract


def mp3_to_wav(mp3_filename: str, wav_filename: str, frame_rate: int) -> None:
    mp3_file = AudioSegment.from_file(file=mp3_filename)
    mp3_file.set_frame_rate(frame_rate).export(wav_filename, format="wav")


@dataclass
class SadTalkerModelBundle:
    paths: Dict[str, str]
    preprocess_model: CropAndExtract
    audio_to_coeff: Audio2Coeff
    animate_from_coeff: AnimateFromCoeff


class SadTalkerService:
    def __init__(self, checkpoint_path: str = "checkpoints", config_path: str = "src/config"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self._model_cache: Dict[Tuple[int, str], SadTalkerModelBundle] = {}
        self._cache_lock = threading.Lock()

        os.environ["TORCH_HOME"] = checkpoint_path

    def get_supported_sizes(self):
        return [256, 512]

    def get_preprocess_family(self, preprocess: str) -> str:
        return "full" if "full" in preprocess else "standard"

    def get_checkpoint_status(self) -> Dict[str, object]:
        sizes = {}
        ready = True

        for size in self.get_supported_sizes():
            required = set()
            for preprocess in ("crop", "full"):
                paths = init_path(
                    self.checkpoint_path,
                    self.config_path,
                    size=size,
                    old_version=False,
                    preprocess=preprocess,
                )
                for path in paths.values():
                    if isinstance(path, str):
                        required.add(path)
            missing = [path for path in sorted(required) if not os.path.exists(path)]
            sizes[str(size)] = {
                "ready": not missing,
                "missing": missing,
            }
            ready = ready and not missing

        return {
            "ready": ready,
            "device": self.device,
            "checkpoint_dir": self.checkpoint_path,
            "supported_sizes": self.get_supported_sizes(),
            "sizes": sizes,
        }

    def _load_models(self, size: int, preprocess: str) -> SadTalkerModelBundle:
        cache_key = (size, self.get_preprocess_family(preprocess))

        with self._cache_lock:
            if cache_key in self._model_cache:
                return self._model_cache[cache_key]

            paths = init_path(
                self.checkpoint_path,
                self.config_path,
                size=size,
                old_version=False,
                preprocess=preprocess,
            )
            bundle = SadTalkerModelBundle(
                paths=paths,
                preprocess_model=CropAndExtract(paths, self.device),
                audio_to_coeff=Audio2Coeff(paths, self.device),
                animate_from_coeff=AnimateFromCoeff(paths, self.device),
            )
            self._model_cache[cache_key] = bundle
            return bundle

    def _prepare_inputs(
        self,
        source_image: str,
        driven_audio: str,
        save_dir: str,
    ) -> Tuple[str, str]:
        input_dir = os.path.join(save_dir, "input")
        os.makedirs(input_dir, exist_ok=True)

        source_dest = os.path.join(input_dir, os.path.basename(source_image))
        shutil.copy2(source_image, source_dest)

        audio_dest = os.path.join(input_dir, os.path.basename(driven_audio))
        shutil.copy2(driven_audio, audio_dest)

        if audio_dest.lower().endswith(".mp3"):
            wav_dest = os.path.splitext(audio_dest)[0] + ".wav"
            mp3_to_wav(audio_dest, wav_dest, 16000)
            audio_dest = wav_dest

        return source_dest, audio_dest

    def run_job(
        self,
        source_image: str,
        driven_audio: str,
        preprocess: str = "crop",
        still_mode: bool = False,
        enhancer: Optional[str] = None,
        batch_size: int = 2,
        size: int = 256,
        pose_style: int = 0,
        exp_scale: float = 1.0,
        result_dir: str = "./results",
        job_id: Optional[str] = None,
    ) -> str:
        bundle = self._load_models(size=size, preprocess=preprocess)

        if job_id is None:
            job_id = str(uuid.uuid4())

        save_dir = os.path.join(result_dir, job_id)
        os.makedirs(save_dir, exist_ok=True)

        pic_path, audio_path = self._prepare_inputs(source_image, driven_audio, save_dir)

        first_frame_dir = os.path.join(save_dir, "first_frame_dir")
        os.makedirs(first_frame_dir, exist_ok=True)

        first_coeff_path, crop_pic_path, crop_info = bundle.preprocess_model.generate(
            pic_path,
            first_frame_dir,
            preprocess,
            True,
            size,
        )
        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        batch = get_data(
            first_coeff_path,
            audio_path,
            self.device,
            ref_eyeblink_coeff_path=None,
            still=still_mode,
            idlemode=False,
            length_of_audio=0,
            use_blink=True,
        )
        coeff_path = bundle.audio_to_coeff.generate(batch, save_dir, pose_style, None)

        data = get_facerender_data(
            coeff_path,
            crop_pic_path,
            first_coeff_path,
            audio_path,
            batch_size,
            still_mode=still_mode,
            preprocess=preprocess,
            size=size,
            expression_scale=exp_scale,
        )
        return_path = bundle.animate_from_coeff.generate(
            data,
            save_dir,
            pic_path,
            crop_info,
            enhancer=enhancer,
            preprocess=preprocess,
            img_size=size,
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        return return_path
