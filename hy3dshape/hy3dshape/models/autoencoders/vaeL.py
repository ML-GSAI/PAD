
import os
from contextlib import contextmanager
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from scipy.stats import truncnorm
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities import rank_zero_only

from ...utils.ema import LitEma
from ...utils.misc import instantiate_from_config, instantiate_non_trainable_model
from ...utils.evaluation.chamfer_distance import chamfer_mesh_point_clouds, chamfer_simple_point_clouds
from ...utils.evaluation.volumn_iou import compute_miou_box3d
from ...utils.evaluation.fscore import fscore_point_clouds
from ...pipelines import export_to_trimesh




class VAEL(pl.LightningModule):
    def __init__(
        self,
        *,
        first_stage_config,
        optimizer_cfg,
        pipeline_cfg=None,
        image_processor_cfg=None,
        lora_config=None,
        ema_config=None,
        first_stage_key: str = "surface",
        cond_stage_key: str = "image",
        scale_by_std: bool = False,
        z_scale_factor: float = 1.0,
        ckpt_path: Optional[str] = None,
        ignore_keys: Union[Tuple[str], List[str]] = (),
        torch_compile: bool = False,
        depth_cond = False,
        val_sample = False,
        query_num = 20000,
        optimize_encoder: bool = False,
        add_kl_loss: bool = False,
        kl_loss_weight: float = 0.001,
        reset_pose_prob: float = 0.0,
    ):
        self.reset_pose_prob = reset_pose_prob
        self.optimize_encoder = optimize_encoder
        self.add_kl_loss = add_kl_loss
        self.kl_loss_weight = kl_loss_weight
        self.val_sample = val_sample
        self.query_num = query_num
        super().__init__()
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key

        # ========= init optimizer config ========= #
        self.optimizer_cfg = optimizer_cfg

        self.first_stage_model = instantiate_from_config(first_stage_config).float()

        if not optimize_encoder:
            for param in self.first_stage_model.encoder.parameters():
                param.requires_grad = True
                print(param.dtype)

            for param in self.first_stage_model.pre_kl.parameters():
                param.requires_grad = True

        # ========= init the model ========= #
        self.model = self.first_stage_model
        self.orig_model = instantiate_from_config(first_stage_config)
        for param in self.orig_model.parameters():
            param.requires_grad = False

        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        # ========= config lora model ========= #
        if lora_config is not None:
            print("using lora")
            from peft import LoraConfig, get_peft_model
            loraconfig = LoraConfig(
                r=lora_config.rank,
                lora_alpha=lora_config.rank,
                lora_dropout=lora_config.dropout,
                target_modules=lora_config.get('target_modules'),
            )
            self.model = get_peft_model(self.model, loraconfig)

        # ========= config ema model ========= #
        self.ema_config = ema_config
        if self.ema_config is not None:
            if self.ema_config.ema_model == 'DSEma':
                # from michelangelo.models.modules.ema_deepspeed import DSEma
                from ..utils.ema_deepspeed import DSEma
                self.model_ema = DSEma(self.model, decay=self.ema_config.ema_decay)
            else:
                self.model_ema = LitEma(self.model, decay=self.ema_config.ema_decay)
            #do not initilize EMA weight from ckpt path, since I need to change moe layers
            if ckpt_path is not None:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # ========= init vae at last to prevent it is overridden by loaded ckpt ========= #

        self.scale_by_std = scale_by_std
        if scale_by_std:
            self.register_buffer("z_scale_factor", torch.tensor(z_scale_factor))
        else:
            self.z_scale_factor = z_scale_factor

        # ========= torch compile to accelerate ========= #
        self.torch_compile = torch_compile
        if self.torch_compile:
            torch.nn.Module.compile(self.first_stage_model)
            print(f'*' * 100)
            print(f'Compile model for acceleration')
            print(f'*' * 100)

    @contextmanager
    def ema_scope(self, context=None):
        if self.ema_config is not None and self.ema_config.get('ema_inference', False):
            self.model_ema.store(self.model)
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.ema_config is not None and self.ema_config.get('ema_inference', False):
                self.model_ema.restore(self.model)
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=()):
        ckpt = torch.load(path, map_location="cpu")
        if 'state_dict' not in ckpt:
            # deepspeed ckpt
            state_dict = {}
            for k in ckpt.keys():
                new_k = k.replace('_forward_module.', '')
                state_dict[new_k] = ckpt[k]
        else:
            state_dict = ckpt["state_dict"]

        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_load_checkpoint(self, checkpoint):
        """
        The pt_model is trained separately, so we already have access to its
        checkpoint and load it separately with `self.set_pt_model`.

        However, the PL Trainer is strict about
        checkpoint loading (not configurable), so it expects the loaded state_dict
        to match exactly the keys in the model state_dict.

        So, when loading the checkpoint, before matching keys, we add all pt_model keys
        from self.state_dict() to the checkpoint state dict, so that they match
        """
        for key in self.state_dict().keys():
            if key.startswith("model_ema") and key not in checkpoint["state_dict"]:
                checkpoint["state_dict"][key] = self.state_dict()[key]

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate

        params_list = []
        trainable_parameters = list(self.first_stage_model.parameters())

        if not self.optimize_encoder:
            filtered_parameters = list(self.first_stage_model.encoder.parameters()) + list(self.first_stage_model.pre_kl.parameters())
            trainable_parameters = list(set(trainable_parameters) - set(filtered_parameters))
        params_list.append({'params': trainable_parameters, 'lr': lr})

        no_decay = ['bias', 'norm.weight', 'norm.bias', 'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias']


        optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=params_list, lr=lr)
        if hasattr(self.optimizer_cfg, 'scheduler'):
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr
            )
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func.schedule),
                "interval": "step",
                "frequency": 1
            }
            schedulers = [scheduler]
        else:
            schedulers = []
        optimizers = [optimizer]

        return optimizers, schedulers

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema_config is not None:
            self.model_ema(self.model)

    def on_train_epoch_start(self) -> None:
        pl.seed_everything(self.trainer.global_rank)
        # pass

    def forward(self, batch):

        B, M, _ = batch[self.first_stage_key].shape # BxMx7
        a, b = -0.25, 0.25
        mu = 0

        random_surface = batch["surface"][:,:,:3]
        offset1 = truncnorm.rvs((a - mu) / 0.005, (b - mu) / 0.005, loc=mu, scale=0.005, size=(B, M, 3))
        offset2 = truncnorm.rvs((a - mu) / 0.05, (b - mu) / 0.05, loc=mu, scale=0.05,  size=(B, M, 3))
        offset1 = torch.from_numpy(offset1).to(random_surface)
        offset2 = torch.from_numpy(offset2).to(random_surface)

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            with torch.no_grad():

                # 1. rotate the input point cloud randomly
                if self.reset_pose_prob > 0.0 and torch.rand(1).item() < self.reset_pose_prob:
                    random_rotation = torch.randn(B, 3, 3).to(latents_dec.device)
                else:
                    random_rotation = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(latents_dec.device)
                u, _, v = torch.linalg.svd(random_rotation)
                random_rotation = torch.matmul(u, v.transpose(-1, -2))
                rotated_surface = batch[self.first_stage_key].clone()
                rotated_surface[:, :, :3] = torch.matmul(rotated_surface[:, :, :3], random_rotation.T)
                rotated_surface[:, :, 3:6] = torch.matmul(rotated_surface[:, :, 3:6], random_rotation.T)

                # 2. find scale from surface and rotated surface
                scale_rot = rotated_surface[:, :, :3].max(dim=1)[0] - rotated_surface[:, :, :3].min(dim=1)[0]
                scale_ori = batch[self.first_stage_key][:, :, :3].max(dim=1)[0] - batch[self.first_stage_key][:, :, :3].min(dim=1)[0]
                scale = torch.max(scale_rot, scale_ori) # B x 3
                scale = scale.max(dim=1)[0] / 1.98      # B
                print("surface scale:", scale, scale.shape)

                # 3. rescale two surface, rotate surface need translation (Assume surface_ori is centered at dataloader)
                surface_ori = batch[self.first_stage_key].clone()
                surface_ori[:, :, :3] = batch[self.first_stage_key][:, :, :3] / scale[:, None, None]
                rotated_surface[:, :, :3] = rotated_surface[:, :, :3] / scale[:, None, None]
                translation = (rotated_surface[:, :, :3].max(dim=1)[0] + rotated_surface[:, :, :3].min(dim=1)[0]) / 2.0
                rotated_surface[:, :, :3] = rotated_surface[:, :, :3] - translation[:, None, :]

                # 4. find query points and sdf values
                latents = self.orig_model.encode(surface_ori, sample_posterior=True)
                latents_dec = self.orig_model.decode(latents)
                random_near_points = torch.cat([
                    random_surface + offset1,
                    random_surface + offset2
                ], dim=1)
                vol_points = (torch.rand(B, M * 2, 3) - 0.5) * 2 * 1.05
                vol_points = vol_points.to(latents_dec)
                all_query_points = torch.cat([
                    vol_points,
                    random_near_points
                ], dim=1)
                all_query_points_idx = torch.randperm(all_query_points.shape[0])[:self.query_num]
                all_query_points = all_query_points[all_query_points_idx]
                sdf_values = self.orig_model.query_sdf(latents_dec, all_query_points)

                
                # 5. find rotated query points (do the same thing as rotated surface)
                all_query_points_rot = torch.matmul(all_query_points, random_rotation.T)
                all_query_points_rot = all_query_points_rot / scale[:, None, None]
                all_query_points_rot = all_query_points_rot - translation[:, None, :]

                
                print("when training, surface range:", torch.max(batch["surface"][0], dim=0), torch.min(batch["surface"][0], dim=0))
        
        with torch.autocast(device_type="cuda", dtype=torch.float32):

            if self.add_kl_loss:
                moments = self.first_stage_model.encode(rotated_surface, sample_posterior=False, ret_moments=True)
                mu, log_var = moments.chunk(2, dim=-1)
                kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1))
                latents = self.first_stage_model.sample_posterior(moments, sample_posterior=True)
                # kl_loss = kl_loss / (B * M)
                # latents = self.z_scale_factor * latents
            else:
                with torch.no_grad():
                    kl_loss = torch.tensor(0.0).to(rotated_surface.device)
                    latents = self.first_stage_model.encode(rotated_surface, sample_posterior=True)

            latents_dec = self.first_stage_model.decode(latents)
            sdf = self.first_stage_model.query_sdf(latents_dec, all_query_points_rot)
            loss = ((sdf - sdf_values) ** 2).mean()

            loss_dict = {
                "total_loss": loss + self.kl_loss_weight * kl_loss,
                "recon_loss": loss,
                "kl_loss": kl_loss,
            }

        return loss_dict

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss_dict_fwd = self.forward(batch)
        split = 'train'
        log_dict = {
            f"{split}/simple": loss_dict_fwd["total_loss"].detach(),
            f"{split}/total_loss": loss_dict_fwd["total_loss"].detach(),
            f"{split}/recon_loss": loss_dict_fwd["recon_loss"].detach(),
            f"{split}/kl_loss": loss_dict_fwd["kl_loss"].detach(),
            f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
        self.log_dict(log_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss_dict_fwd["total_loss"]

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        loss_dict_fwd = self.forward(batch)
        split = 'val'
        cd_dict_mean = {}

        if self.val_sample:
            from pathlib import Path
            import numpy as np
            save_dir_root = Path(self.logger.log_dir)
            save_dir = save_dir_root / "val_samples"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_dir_step = save_dir / f"step_{self.global_step}"
            save_dir_step.mkdir(parents=True, exist_ok=True)
            
            # get distributed world idx
            world_idx = self.trainer.global_rank

            

            objects_pred = self.sample(batch, output_type='trimesh')

            # ***************** random rotate input point cloud  **************************

            CD_SAMPLES = 300000

            # first sample orig objects as gts
            objects_orig = self.sample(batch, output_type='trimesh', use_orig=True)
            pts_sample_orig = []
            pts_rotated_orig = []
            for obj in objects_orig:
                pts = obj.sample(CD_SAMPLES).astype(np.float32)
                pts_sample_orig.append(pts)


            # rotate input point cloud to rotated
            B = batch[self.first_stage_key].shape[0]
            random_rotation = torch.randn(B, 3, 3).to(batch[self.first_stage_key].device)
            u, _, v = torch.linalg.svd(random_rotation)
            random_rotation = torch.matmul(u, v.transpose(-1, -2))
            rotated_surface = batch[self.first_stage_key].clone().float()
            rotated_surface[:, :, :3] = torch.matmul(rotated_surface[:, :, :3], random_rotation)
            rotated_surface[:, :, 3:6] = torch.matmul(rotated_surface[:, :, 3:6], random_rotation)
            for i in range(len(pts_sample_orig)):
                pts_rotated_orig.append(np.dot(pts_sample_orig[i], random_rotation[i].cpu().numpy()))

            # normalize to unit cube -1 to 1
            scale = rotated_surface[:, :, :3].max(dim=1)[0] - rotated_surface[:, :, :3].min(dim=1)[0]
            scale = scale.max(dim=1)[0] / 1.98
            translation = (rotated_surface[:, :, :3].max(dim=1)[0] + rotated_surface[:, :, :3].min(dim=1)[0]) / 2.0
            rotated_surface[:, :, :3] = (rotated_surface[:, :, :3] - translation[:, None, :]) / scale[:, None, None]
            for i in range(len(pts_rotated_orig)):
                pts_rotated_orig[i] = (pts_rotated_orig[i] - translation[i].cpu().numpy()) / scale[i].cpu().numpy()

            # sample rotated objects as inputs
            batch_rotated = {
                self.first_stage_key: rotated_surface
            }
            objects_rotated = self.sample(batch_rotated, output_type='trimesh')
            pts_pred_sample_rotated = []
            for obj in objects_rotated:
                pts = obj.sample(CD_SAMPLES).astype(np.float32)
                pts_pred_sample_rotated.append(pts)

            # compute cd between orig and rotated objects
            cd_rotated = []
            for i in range(len(pts_sample_orig)):
                    cd = chamfer_simple_point_clouds(pts_pred_sample_rotated[i], pts_rotated_orig[i])
                    cd_dict = {}
                    cd_dict["rotated_predorig_cd_l2"] = cd["cd_l2"]

                    
                    cd = chamfer_simple_point_clouds(pts_pred_sample_rotated[i], rotated_surface[i][:,:3].cpu().numpy())
                    cd_dict["rotated_predinput_cd_l2"] = cd["cd_l2"]

                    pts_pred = objects_pred[i].sample(CD_SAMPLES).astype(np.float32)
                    cd2 = chamfer_simple_point_clouds(pts_pred, pts_sample_orig[i])
                    cd_dict[f"reset_predorig_cd_l2"] = cd2["cd_l2"]

                    cd = chamfer_simple_point_clouds(pts_pred, batch[self.first_stage_key][i][:,:3].float().cpu().numpy())
                    cd_dict["reset_predinput_cd_l2"] = cd["cd_l2"]

                    cd_rotated.append(cd_dict)

            for key in cd_rotated[0].keys():
                cd_dict_mean[key] = sum([cd[key] for cd in cd_rotated]) / len(cd_rotated)


            # *******************************************************


            objects_gt = batch['surface']

            objects_gt = objects_gt.to(torch.float32)
            # object_pred = [pred.to(torch.float32) for pred in objects_pred]

            cds = []
            fscores = []
            ious = []

            for i in range(len(batch['surface'])):
                object_pred = objects_pred[i]
                save_file_name = f"idx_{batch_idx}_device_{world_idx}_{i}.glb"
                save_file_path = save_dir_step / save_file_name
                print(save_file_path)
                img = batch['image'][i]
                save_img_path = save_dir_step / f"idx_{batch_idx}_device_{world_idx}_{i}_image.png"
                import torchvision
                torchvision.utils.save_image(img, str(save_img_path))
                if object_pred is not None:
                    object_pred.export(str(save_file_path))
                
                object_rot = objects_rotated[i]
                save_file_name_rot = f"idx_{batch_idx}_device_{world_idx}_{i}_rotated.glb"
                save_file_path_rot = save_dir_step / save_file_name_rot
                if object_rot is not None:
                    object_rot.export(str(save_file_path_rot))

            for ind in range(len(objects_pred)):
                object_pred = objects_pred[ind]
                if object_pred is None:
                    continue
                pts_pred = object_pred.sample(100000).astype(np.float32)
                try:
                    cd = chamfer_mesh_point_clouds(object_pred, objects_gt[ind,:,:3].cpu().numpy())
                    fscore = fscore_point_clouds(pts_pred, objects_gt[ind,:,:3].cpu().numpy(), threshold=0.05)
                    iou = compute_miou_box3d(pts_pred, objects_gt[ind,:,:3].cpu().numpy())
                    metrics = {}
                    metrics.update(cd)
                    metrics.update(fscore)
                    metrics.update(iou)
                    cds.append(metrics)
                except:
                    continue

            for key in cds[0].keys():
                cd_dict_mean[key] = sum([cd[key] for cd in cds]) / len(cds)



        log_dict = {
            f"{split}/simple": loss_dict_fwd["total_loss"].detach(),
            f"{split}/total_loss": loss_dict_fwd["total_loss"].detach(),
            f"{split}/recon_loss": loss_dict_fwd["recon_loss"].detach(),
            f"{split}/kl_loss": loss_dict_fwd["kl_loss"].detach(),
        }

        log_dict.update({f"{split}/{k}": v for k, v in cd_dict_mean.items()})
        self.log_dict(log_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss_dict_fwd["total_loss"]

    @torch.no_grad()
    def sample(self, batch, output_type='trimesh', use_orig = False, **kwargs):

        generator = torch.Generator().manual_seed(0)
        if use_orig:
            model = self.orig_model
        else:
            model = self.first_stage_model
        with self.ema_scope("Sample"):
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                try:
                    model.eval()
                    latents = model.encode(batch[self.first_stage_key], sample_posterior=True)
                    latents = model.decode(latents)
                    outputs = model.latents2mesh(
                        latents,
                        output_type=output_type,
                        bounds=1.01,
                        mc_level=0.0,
                        num_chunks=20000,
                        octree_resolution=256,
                        mc_algo='mc',
                        enable_pbar=True,
                    )
                    outputs = export_to_trimesh(outputs)
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Unexpected {e=}, {type(e)=}")
                    with open("error.txt", "a") as f:
                        f.write(str(e))
                        f.write(traceback.format_exc())
                        f.write("\n")
                    outputs = [None]

        return outputs
