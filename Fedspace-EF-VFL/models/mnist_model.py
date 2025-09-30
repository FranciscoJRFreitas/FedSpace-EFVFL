import torch
import torch.nn as nn
from models.lightning_splitnn import SplitNN
from models.compressors import CompressionModule
import math

class RepresentationModel(nn.Module):
    def __init__(self, input_size, cut_size, compressor, compression_parameter, compression_type, num_samples):
        super().__init__()
        self.fc = nn.Linear(input_size, cut_size)
        self.relu = nn.ReLU()
        self.compression_module = CompressionModule(compressor, compression_parameter, compression_type, num_samples, cut_size)

    def forward(self, x, apply_compression=False, indices=None, epoch=None):
        x = self.relu(self.fc(x))
        return self.compression_module(x, apply_compression, indices, epoch)

class FusionModel(nn.Module):
    def __init__(self, cut_size, num_classes, aggregation_mechanism, num_clients):
        super().__init__()
        self.num_classes = num_classes
        self.aggregation_mechanism = aggregation_mechanism
        fusion_input_size = cut_size * num_clients if aggregation_mechanism == "conc" else cut_size
        self.fc = nn.Linear(fusion_input_size, num_classes)

    def forward(self, x):
        if self.aggregation_mechanism == "conc":
            B = x[0].shape[0]
            for j, xi in enumerate(x):
                assert xi.shape[0] == B, f"Batch size mismatch in conc at part {j}: {xi.shape} vs {B}"
            aggregated_x = torch.cat(x, dim=1)

        elif self.aggregation_mechanism == "mean":
            stacked = torch.stack(x, dim=0)
            mask = (stacked.abs().sum(dim=-1) > 0).float()
            denom = mask.sum(dim=0).clamp_min(1.0).unsqueeze(-1)
            aggregated_x = (stacked * mask.unsqueeze(-1)).sum(dim=0) / denom

        elif self.aggregation_mechanism == "sum":
            aggregated_x = torch.stack(x).sum(dim=0)

        return self.fc(aggregated_x)


class ShallowSplitNN(SplitNN):
    def __init__(self, input_size, num_clients, cut_size, aggregation_mechanism, num_classes, private_labels,
             lr, momentum, weight_decay, compressor, compression_parameter, compression_type,
             optimizer, eta_min_ratio, scheduler, num_epochs,
             compute_grad_sqd_norm, num_samples, batch_size,
             connectivity_schedule=None, fedspace_scheduler=None, partial_participation: bool = True, agg_mode: str = "sync", buffer_M: int = 96):

        assert num_clients == 149, "This setup assumes exactly 149 clients, but got {}".format(num_clients)
        self.slice_sizes = [6]*39 + [5]*110
        assert sum(self.slice_sizes) == input_size, "Slice sizes must sum to total input size"

        self.cut_size = cut_size

        representation_models = nn.ModuleList([
            RepresentationModel(slice_size, cut_size, compressor,
                                compression_parameter, compression_type, num_samples)
            for slice_size in self.slice_sizes
        ])

        # local_input_size = input_size // num_clients
        # representation_model_parameters = local_input_size, cut_size, compressor, compression_parameter, compression_type, num_samples
        # representation_models = nn.ModuleList([RepresentationModel(*representation_model_parameters) for _ in range(num_clients)])
        

        fusion_model = FusionModel(cut_size, num_classes, aggregation_mechanism, num_clients)

        super().__init__(representation_models, fusion_model, lr, momentum, weight_decay,
                        optimizer, eta_min_ratio, scheduler, num_epochs, private_labels,
                        batch_size, compute_grad_sqd_norm, connectivity_schedule,
                        fedspace_scheduler, partial_participation, agg_mode=agg_mode, buffer_M=buffer_M)

    
    def get_feature_block(self, x: torch.Tensor, i: int) -> torch.Tensor:
        B, C, H, W = x.shape
        N_pix = H * W
        flat = x.reshape(B, C, N_pix)

        start = sum(self.slice_sizes[:i])
        end   = start + self.slice_sizes[i]

        patch = flat[:, :, start:end]
        return patch.reshape(B, -1)

    # def get_feature_block(self, x, i):
    #     """Get the quadrant i of the input image x."""
    #     _, _, H, W = x.shape
    #     if not 0 <= i <= 3:
    #         raise ValueError("Invalid index i: Choose i from 0, 1, 2, or 3.")
    #     if self.num_clients != 4:
    #         raise ValueError("Assuming 4 clients, each holding a quadrant.")

    #     if i == 0:  # Top-left
    #         quadrant = x[:, :, :H//2, :W//2]
    #     elif i == 1:  # Top-right
    #         quadrant = x[:, :, :H//2, W//2:]
    #     elif i == 2:  # Bottom-left
    #         quadrant = x[:, :, H//2:, :W//2]
    #     elif i == 3:  # Bottom-right
    #         quadrant = x[:, :, H//2:, W//2:]

    #     flat_quadrant = quadrant.reshape(quadrant.size(0), -1)
    #     return flat_quadrant