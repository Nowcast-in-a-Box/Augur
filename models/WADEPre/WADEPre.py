import torch
from lightning.pytorch import LightningModule
from torch.nn import functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure


from models.WADEPre.Approximation import ApproximationNetwork
from models.WADEPre.Detail import DetailNetwork
from models.WADEPre.Refiner import RefineMixer

from utils.wavelet_transform import WaveletTransform, WaveletCoeffDict


from utils.metrics import log_loss, compute_loss

from utils.muon import MuonWithAuxAdam

from torchmetrics import MeanMetric, MetricCollection

from utils.statistic import print_result
from rich.console import Console

class WADEPre(LightningModule):
    def __init__(
        self,
        # general params
        timesteps: int,
        spatial_size: int,
        dropout_rate: float = 0.1,
        # detail network params
        detail_idr_dim: int = 32,
        detail_feature_channel: int = 64,
        detail_layer_channels: list = [64, 128, 256],
        detail_num_blocks: int = 4,
        # approximation network params
        approx_hidden_size: int = 128,
        approx_cells: int = 3,
        # refine mixer params
        refine_hidden_dim: int = 128,
        # wavelet params
        wavelet_name: str = "bior2.4",
        wavelet_level: int = 3,
        # model params
        lr: float = 1e-3,
        # loss
        loss_a_weight: float = 1.0,
        loss_a_constant_weight: float = 0.15,
        loss_a_stop_step: int = 5000,
        loss_d_weight: float = 1.0,
        loss_recon_mean_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.a_weight = loss_a_weight
        
        self.a_weight_decay = (loss_a_weight - loss_a_constant_weight) / loss_a_stop_step
        self.loss_a_constant_weight = loss_a_constant_weight
        
        self.loss_a_stop_step = loss_a_stop_step
        self.d_weight = loss_d_weight
        self.loss_recon_mean_weight= loss_recon_mean_weight

        self.detail_network = DetailNetwork(
            fpn_time=timesteps,
            idr_dim=detail_idr_dim,
            feature_channel=detail_feature_channel,
            layer_channels=detail_layer_channels,
            num_blocks=(
                detail_num_blocks
                if isinstance(detail_num_blocks, (list, tuple))
                else [detail_num_blocks] * len(detail_layer_channels)
            ),
            dropout_rate=dropout_rate
        )

        self.approx_network = ApproximationNetwork(hidden_size=approx_hidden_size, 
                                                   timesteps=timesteps,
                                                   cell_numbers=approx_cells,
                                                   dropout_rate=dropout_rate)

        self.refine_mixer = RefineMixer(
            time_steps=timesteps,
            hidden_dim=refine_hidden_dim,
            dropout_rate=dropout_rate
        )

        self.wavelet_transform = WaveletTransform(
            wavelet=wavelet_name, level=wavelet_level, mode="reflect"
        )

        self.lr = lr

        
        self.init_model()

    def init_model(self) -> None:
        
        with torch.no_grad():
        
            # dummy run
            
            x = torch.randn(
                2,
                self.hparams.timesteps,
                self.hparams.spatial_size,
                self.hparams.spatial_size,
            )
            self.detail_network.dummy_run(data=x, wavelet=self.wavelet_transform)
            self.approx_network.dummy_run(data=x, wavelet=self.wavelet_transform)

    def forward(self, x: torch.Tensor):
        torch.save(self.state_dict(), "script.pt")
        # print(f"[Probe 0 - X]: {x.float().sum().item():.7f}")
        d_reconstruction, d_coeff = self.detail_network.forward(
            x, wavelet=self.wavelet_transform
        )
        
        # print(f"First: {self.detail_network.temporal_mlp_before[0].weight.sum().item():.10f}")
        
        a_reconstruction, a_coeff = self.approx_network.forward(
            x, wavelet=self.wavelet_transform
        )
        
        # print(f"[Probe 1 - Details]: {d_reconstruction.float().sum().item():.7f}")
        # print(f"[Probe 1 - Approximation]: {a_reconstruction.float().sum().item():.7f}")

        # a_coeff is the coeffs dict returned by ApproximationNetwork; take its A
        ad_coeff: WaveletCoeffDict = {"A": a_coeff["A"]}

        for l in range(1, self.wavelet_transform.level + 1):
            level_key = f"D{l}"
            level_details = d_coeff[level_key]  # Tensor shaped (B, T, 3, H, W)

            # Ensure it's 3 detail channels stacked along dim=2
            if level_details.shape[2] != 3:
                raise RuntimeError(
                    f"Expected 3 detail channels at level {l}, got shape {level_details.shape}"
                )

            # Already (B, T, 3, H, W) — stack over channel 2 (already stacked)
            ad_coeff[level_key] = level_details

        ad_reconstruction = self.wavelet_transform.reverse(ad_coeff)
        # print(f"[Probe 2 - AD_Reconstruction]: {ad_reconstruction.float().sum().item():.7f}")

        refined_out = self.refine_mixer.forward(
            AD_guide=ad_reconstruction,
            A_guide=a_reconstruction,
            D_guide=d_reconstruction,
            last_frame=x[:, -1:, :, :]
        )
        # print(f"[Probe 3 - Refined]: {refined_out.float().sum().item():.7f}")

        return refined_out, {
            "d_rec": d_reconstruction,
            "a_rec": a_reconstruction,
            "ad_rec": ad_reconstruction,
            "refined_out": refined_out,
            "d_coeff": d_coeff,
            "a_coeff": a_coeff,
            "ad_coeff": ad_coeff,
        }

    def compute_loss(
        self, x: dict, truth: torch.Tensor, stage: str = "train"
    ) -> torch.Tensor:

        truth_wave: WaveletCoeffDict = self.wavelet_transform.transform(truth)

        if self.global_step < self.loss_a_stop_step:
            a_weight = self.a_weight - self.global_step * self.a_weight_decay
        else:
            a_weight = self.loss_a_constant_weight

        
        main_recon = F.smooth_l1_loss(x["refined_out"], truth)
        # main_recon = self.facl.compute_facl(x["refined_out"], truth, current_step=self.global_step)


        # A coeff loss
        # a_coeff_loss = F.mse_loss(x["a_coeff"]["A"], truth_wave["A"])
        a_coeff_loss = F.l1_loss(x["a_coeff"]["A"], truth_wave["A"])

        # D coeff loss
        d_coeff_loss = 0.0
        for l in range(1, self.wavelet_transform.level + 1):
            d_coeff_loss += F.mse_loss(x["d_coeff"][f"D{l}"], truth_wave[f"D{l}"]) * (
                1.0 / 2**l
            )

        
        # Reconstruction mean loss
        reconstruction_mean = (x["ad_rec"] + x["a_rec"] + x["d_rec"]) / 3
        reconstruction_mean_loss = F.mse_loss(reconstruction_mean, truth)
        
        
        total_loss = (
            main_recon + a_weight * a_coeff_loss + self.d_weight * d_coeff_loss + self.loss_recon_mean_weight * reconstruction_mean_loss
        )
                
        self.log(f"{stage}_recon", F.mse_loss(x["refined_out"], truth), sync_dist=True, prog_bar=False)
        self.log(f"{stage}_recon_mean", reconstruction_mean_loss, sync_dist=True, prog_bar=False)
        self.log(f"{stage}_a_coeff", a_coeff_loss, sync_dist=True, prog_bar=False)
        self.log(f"{stage}_a_weight", a_weight, sync_dist=True, prog_bar=False)
        self.log(f"{stage}_d_coeff", d_coeff_loss, sync_dist=True, prog_bar=False)
        self.log(f"{stage}_loss", total_loss, sync_dist=True, prog_bar=False)

        return total_loss

    def training_step(self, batch):
        data = batch["sequence"]
        target = batch["target"]

        output, details = self.forward(data)

        loss = self.compute_loss(details, target, stage="train")
        
        log_loss(
            pred=output,
            truth=target,
            stage="trian",
            lightning_module=self
        )

        return loss

    def validation_step(self, batch):
        data = batch["sequence"]
        target = batch["target"]

        output, details = self.forward(data)

        loss = self.compute_loss(details, target, stage="val")
        log_loss(
            pred=output,
            truth=target,
            stage="val",
            lightning_module=self
        )

        return loss

    def configure_optimizers(self):
        
        
        # optimizer = torch.optim.AdamW(
        #     params=self.parameters(),
        #     lr=self.lr,
        #     weight_decay=0.01,
        #     eps=5e-7,
        #     betas=(0.9, 0.995),
        # )


        
        
        # Muon optimizer
        
        muon_param, adamw_param = self._get_params_for_muon()
        
        optimizer = MuonWithAuxAdam(
            [
                dict(
                    params=muon_param,
                    use_muon=True,
                    lr=5e-3,
                    weight_decay=0.01
                ),
                dict(
                    params=adamw_param,
                    use_muon=False,
                    lr=self.lr,
                    betas=(0.9, 0.995),
                    weight_decay=0.0
                )
            ]
        )
        
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.trainer.max_epochs,
        )
        

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }

    
    def _get_params_for_muon(self):
        
        muon_params = []
        adamw_params = []
        
        param_keywords = [
            "bias",
            ".norm",
            "bn",
            # meaning
            "input_adapters",
            "temporal_mlp",
            "toplayer",
            "lateral_layers",
            "smooth_layers",
            "a_mask_conv",
            "out_mixer.2",
            "decoder_CNN",
        ]
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
                
            if any(k in name for k in param_keywords):
                adamw_params.append(param)
            elif param.ndim >= 2:
                muon_params.append(param)
            else:
                adamw_params.append(param)
        
        assert len(list(self.parameters())) == len(muon_params) + len(adamw_params)
        
        return muon_params, adamw_params
    
    
    def on_load_checkpoint(self, checkpoint):
        
        state_dict = checkpoint.pop("state_dict", None)
        self.setup("predict")
        
        if state_dict is not None:
            checkpoint["state_dict"] = state_dict
        

    
    def predict_step(self, batch, batch_idx):
        
        sequence = batch["sequence"]
        target = batch["target"]
        
        
        

        output, details = self.forward(x=sequence)
            

        # metrics = compute_loss(pred=output, truth=target)
        


        for idx, lead in enumerate(self.lead_times):
            lead_output = output[:, idx:idx+1, ...]
            lead_target = target[:, idx: idx+1, ...]
            
        

            metrics = compute_loss(
                pred=lead_output,
                truth=lead_target
            )

            collection = self.metric_collection_bbdf[f"lead_{lead}"]
            
            for metric_name, metric_value in metrics.items():

                collection[metric_name].update(metric_value)
        
        return metrics

    def on_predict_epoch_start(self):
        
        self.lead_times = (10, 20, 30 ,40 ,50 ,60)
        
        self.metric_collection_bbdf = {
            f"lead_{h}": MetricCollection({
                "rmse": MeanMetric(),
                "psd": MeanMetric(),
                "csi_mean": MeanMetric(),
                "csi_181": MeanMetric(),
                "csi_219": MeanMetric(),
                "ssim": MeanMetric()
            }).to(self.device)
            for h in self.lead_times
        }
    
    def on_predict_epoch_end(self):
        METRICS_NAME = ("rmse", "psd", "ssim", "csi_mean", "csi_181", "csi_219")
        
        table = print_result(metric_collection=self.metric_collection_bbdf, 
                             metric_names=METRICS_NAME, lead_times=self.lead_times)
        console = Console()
        console.print(table)