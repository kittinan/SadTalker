from src.sadtalker_service import SadTalkerService


class SadTalker:

    def __init__(self, checkpoint_path='checkpoints', config_path='src/config', lazy_load=False):
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.service = SadTalkerService(checkpoint_path=checkpoint_path, config_path=config_path)


    def test(self, source_image, driven_audio, preprocess='crop',
        still_mode=False,  use_enhancer=False, batch_size=1, size=256,
        pose_style = 0, exp_scale=1.0,
        use_ref_video = False,
        ref_video = None,
        ref_info = None,
        use_idle_mode = False,
        length_of_audio = 0, use_blink=True,
        result_dir='./results/'):
        if use_ref_video or use_idle_mode:
            raise NotImplementedError("Reference video and idle mode remain unsupported in the shared service path")

        return self.service.run_job(
            source_image=source_image,
            driven_audio=driven_audio,
            preprocess=preprocess,
            still_mode=still_mode,
            enhancer='gfpgan' if use_enhancer else None,
            batch_size=batch_size,
            size=size,
            pose_style=pose_style,
            exp_scale=exp_scale,
            result_dir=result_dir,
        )
