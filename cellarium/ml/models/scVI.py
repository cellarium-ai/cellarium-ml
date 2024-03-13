import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence as kl
from torch.distributions import Normal, Poisson, NegativeBinomial

from cellarium.ml.models.encoder import Encoder
from cellarium.ml.models.decoder import DecoderSCVI

from cellarium.ml.models.model import CellariumModel

from typing import Literal

def one_hot(index: torch.Tensor, n_cat: int) -> torch.Tensor:

    """One hot a tensor of categories."""

    onehot = torch.zeros(index.size(0), n_cat, device=index.device)
    onehot.scatter_(1, index.type(torch.long), 1)
    return onehot.type(torch.float32)

class scVI(CellariumModel):
    def __init__(
            self,
            n_input: int,
            n_labels: int = 0,
            n_hidden: int = 128,
            n_latent: int = 20,
            n_batch: int = 1,
            n_layers_encoder: int = 4,
            dropout_rate: float = 0.25,

            dispersion: str = "gene",
            log_variational: bool = True,
            gene_likelihood: Literal["nb", "poisson"] = "nb",
            latent_distribution: str = "normal",
            encode_covariates: bool = False,
            deeply_inject_covariates: bool = True,
            use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
            bias: bool = False,
    ):
        super().__init__()
        self.n_input = n_input
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.n_batch = n_batch
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', "
                " 'gene-label', 'gene-cell'], but input was "
                "{}".format(self.dispersion)
            )

        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm in ["encoder", "both"],
            use_layer_norm=False,
        )

        self.decoder = DecoderSCVI(
            n_latent,
            n_input,
            use_batch_norm=use_batch_norm in ["decoder", "both"],
            use_layer_norm=False,
            bias=bias,
        )

    def inference(self, x, n_samples=1):
        """
        High level inference method.
        Runs the inference (encoder) model.
        """
        x_ = x
        library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
        encoder_input = x_

        qz_m, qz_v, z = self.z_encoder(encoder_input)

        if n_samples > 1:
            qz_m = qz_m.unsqueeze(0).expand((n_samples, qz_m.size(0), qz_m.size(1)))
            qz_v = qz_v.unsqueeze(0).expand((n_samples, qz_v.size(0), qz_v.size(1)))
            untran_z = Normal(qz_m, qz_v.sqrt()).sample()
            z = self.z_encoder.z_transformation(untran_z)
            library = library.unsqueeze(0).expand(
                (n_samples, library.size(0), library.size(1))
            )

        outputs = dict(
            z=z,
            qz_m=qz_m,
            qz_v=qz_v,
            library=library,
        )

        return outputs

    def generative(self, z, library, y=None):
        """Runs the generative model."""
        # Likelihood distribution

        decoder_input = z

        categorical_input = tuple()
        size_factor = library

        px_scale, px_r, px_rate = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            *categorical_input,
        )
        if self.dispersion == "gene-label":
            # px_r gets transposed - last dimension is nb genes
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate)

        # Priors
        pl = None
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        return dict(px=px, pl=pl, pz=pz)

    def forward(self, x: torch.Tensor):

        inference_outputs = self.inference(x)
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        # KL divergence
        kl_divergence_z = kl(
            Normal(qz_m, torch.sqrt(qz_v)),
            Normal(mean, scale)).sum(
            dim=1
        )

        generative_outputs = self.generative(inference_outputs["z"], inference_outputs["library"])
        # reconstruction loss
        rec_loss = -generative_outputs["px"].log_prob(x).sum(-1)

        loss = torch.mean(
            rec_loss * kl_divergence_z
        )

        return {"loss": loss}