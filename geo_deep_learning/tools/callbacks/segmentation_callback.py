import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import ListedColormap
from lightning.pytorch.callbacks import ModelCheckpoint

class SegmentationCallback(ModelCheckpoint):
    
    def __init__(self, max_samples: int = 3, class_colors = None, save_dir: str = "visualization_logs",  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_samples = max_samples
        self.class_colors = class_colors
        self.current_batch = None
        self.current_outputs = None
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self._setup_colormap()
    
    def on_validation_batch_end(self,
                                trainer,
                                pl_module,
                                outputs,
                                batch,
                                batch_idx,
                                dataloader_idx=0):
        if trainer.is_global_zero:
            self.current_batch = batch
            self.current_outputs = outputs
    
    def _save_checkpoint(self, trainer, filepath):
        super()._save_checkpoint(trainer, filepath)
        if trainer.is_global_zero:
            self._log_visualizations(trainer)
        
    def _setup_colormap(self):
        if self.class_colors is None:
            # Default colormap for unknown number of classes
            self.cmap = plt.get_cmap('tab20')
        else:
            self.cmap = ListedColormap(self.class_colors)
            
    def _process_image(self, image, mean, std, data_type_max):
        image = image.permute(1, 2, 0).cpu().numpy()    
        num_channels = image.shape[-1]
        
        if num_channels > 3:
            image = image[..., :3]
            num_channels = 3
        if mean is not None and std is not None:
            mean = np.array(mean[:num_channels]).reshape(1, 1, num_channels)
            std = np.array(std[:num_channels]).reshape(1, 1, num_channels)
            image = image * std + mean
        
        image = (image * data_type_max).astype(np.uint8)
        
        return image
    
    def _log_visualizations(self, trainer):
        mean = trainer.datamodule.mean
        std = trainer.datamodule.std
        data_type_max = trainer.datamodule.data_type_max
        if self.current_batch is not None and self.current_outputs is not None:
            image_batch = self.current_batch["image"]
            mask_batch = self.current_batch["mask"]
            batch_image_name = self.current_batch["image_name"]
            batch_mask_name = self.current_batch["mask_name"]
            batch_size = image_batch.shape[0]
            N = min(self.max_samples, batch_size)
            num_classes = mask_batch.max().item() + 1 if self.class_colors is None else len(self.class_colors)
            epoch_dir = self.save_dir / f"epoch_{trainer.current_epoch}"
            epoch_dir.mkdir(exist_ok=True)
            
            try:
                fig, axes = plt.subplots(N, 3, figsize=(15, 5 * N))
                axes = axes.reshape(N, 3) if N > 1 else axes.reshape(1, 3)
                
                for i in range(N):
                    image = image_batch[i]
                    mask = mask_batch[i]
                    image_name = batch_image_name[i]
                    mask_name = batch_mask_name[i]
                    output = self.current_outputs[i]
                    print(f"\nImage Shape_0-1: {image.shape}\n Image Max_0-1: {image.max()}\n Image Min_0-1: {image.min()}\n Image Mean_0-1: {image.mean()}\n")
                    image = self._process_image(image, mean, std, data_type_max)
                    print(f"\nImage Shape_0-255: {image.shape}\n Image Max_0-255: {image.max()}\n Image Min_0-255: {image.min()}\n Image Mean_0-255: {image.mean()}\n")
                    mask = mask.squeeze(0).long().cpu().numpy()
                    output = output.cpu().numpy()
                    
                    sample_dir = epoch_dir / f"sample_{i}"
                    sample_dir.mkdir(exist_ok=True)
                    
                    plt.imsave(sample_dir / f"{image_name}_image.png", image)
                    plt.imsave(sample_dir / f"{mask_name}_mask.png", mask, cmap=self.cmap, vmin=0, vmax=num_classes-1)
                    plt.imsave(sample_dir / f"{image_name}_prediction.png", output, cmap=self.cmap, vmin=0, vmax=num_classes-1)   
                    
                    ax_image, ax_mask, ax_output = axes[i]
                    
                    ax_image.imshow(image)
                    ax_image.set_title("Image")
                    ax_image.axis("off")
                    ax_image.text(0.5, -0.1, f"{image_name}", 
                            transform=ax_image.transAxes,
                            ha='center', va='top',
                            wrap=True)
                    
                    ax_mask.imshow(mask, cmap=self.cmap, vmin=0, vmax=num_classes-1)
                    ax_mask.set_title("Mask")
                    ax_mask.axis("off")
                    # ax_mask.text(0.5, -0.1, f"{mask_name}",
                    #          transform=ax_mask.transAxes,
                    #          ha='center', va='top',
                    #          wrap=True)
                    
                    ax_output.imshow(output, cmap=self.cmap, vmin=0, vmax=num_classes-1)
                    ax_output.set_title('Output')
                    ax_output.axis("off")
                
                plt.tight_layout()
                artifact_file = f"val/predictions_epoch_{trainer.current_epoch}.png"
                trainer.logger.experiment.log_figure(figure=fig, 
                                                    artifact_file = artifact_file,
                                                    run_id=trainer.logger.run_id)
                fig.savefig(epoch_dir / "combined_visualization.png")
                plt.close(fig)
            except Exception as e:
                print(f"Error in visualization: {str(e)}")
            
            finally:
                self.current_batch = None
                self.current_outputs = None