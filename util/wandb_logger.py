"""
Weights & Biases logging (main process only), with the add_scalar / add_image interface used by the training loop.
Every value is logged against "epoch" (fractional during an epoch), so training curves, sample grids and
evaluation metrics share one x-axis. The run id is stored in the output folder, so a resumed training run
(--resume) continues the same W&B run.
"""
import os


class WandbLogger:
    RUN_ID_FILE = 'wandb_run_id.txt'

    def __init__(self, args, project, entity=None, name=None, mode='online', resume=False):
        try:
            import wandb
        except ImportError as e:
            raise ImportError('wandb is not installed: pip install wandb, or run with --no_wandb') from e
        self.wandb = wandb

        run_id, id_file = None, None
        if args.output_dir:
            id_file = os.path.join(args.output_dir, self.RUN_ID_FILE)
            if resume and os.path.exists(id_file):
                with open(id_file) as f:
                    run_id = f.read().strip() or None
        self.run = wandb.init(project=project, entity=entity, name=name or os.path.basename(os.path.normpath(args.output_dir)),
                              id=run_id, resume='allow' if run_id else None, mode=mode,
                              dir=args.output_dir or None, config=vars(args))
        if id_file:
            with open(id_file, 'w') as f:
                f.write(self.run.id)
        self.run.define_metric('epoch')
        self.run.define_metric('*', step_metric='epoch')
        self.log_dir = self.run.dir
        print('W&B run {} ({}): {}'.format(self.run.name, mode, getattr(self.run, 'url', None) or self.run.dir))

    def log(self, values, epoch):
        self.run.log({**values, 'epoch': epoch})

    def add_scalar(self, tag, value, epoch):
        self.log({tag: value}, epoch)

    def add_image(self, tag, image, epoch, caption=None):
        """image: [3, H, W] tensor in [0, 1] (e.g. a torchvision grid)"""
        array = (image.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype('uint8')
        self.log({tag: self.wandb.Image(array, caption=caption)}, epoch)

    def flush(self):
        pass  # wandb uploads in the background

    def finish(self):
        self.run.finish()
