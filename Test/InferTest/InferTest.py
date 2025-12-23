import os
import os.path as osp
import pickle
import haiku as hk
import jmp
import jax
from jax import config
import jax.numpy as jnp
from jax_sph.jax_md import space
import numpy as np
from omegaconf import OmegaConf
from typing import Callable, Dict, Optional, Tuple, Type, Union
from e3nn_jax import Irreps

from lagrangebench import Trainer, infer, models
from lagrangebench.case_setup import case_builder
from lagrangebench.data import H5Dataset
from lagrangebench.defaults import check_cfg
from lagrangebench.evaluate import averaged_metrics
from lagrangebench.models.utils import node_irreps
from lagrangebench.utils import NodeType

# 기존에 사용하신 함수들을 import했다고 가정
# from your_project import setup_data, case_builder, setup_model, check_cfg

def setup_data(cfg) -> Tuple[H5Dataset, H5Dataset, H5Dataset]:
    dataset_path = cfg.dataset.src
    dataset_name = cfg.dataset.name
    ckp_dir = cfg.logging.ckp_dir
    rollout_dir = cfg.eval.rollout_dir
    input_seq_length = cfg.model.input_seq_length
    n_rollout_steps = cfg.eval.n_rollout_steps
    nl_backend = cfg.neighbors.backend

    if not osp.isabs(dataset_path):
        dataset_path = osp.join(os.getcwd(), dataset_path)

    if ckp_dir is not None:
        os.makedirs(ckp_dir, exist_ok=True)
    if rollout_dir is not None:
        os.makedirs(rollout_dir, exist_ok=True)

    # dataloader
    data_train = H5Dataset(
        "train",
        dataset_path=dataset_path,
        name=dataset_name,
        input_seq_length=input_seq_length,
        extra_seq_length=cfg.train.pushforward.unrolls[-1],
        nl_backend=nl_backend,
    )
    data_valid = H5Dataset(
        "valid",
        dataset_path=dataset_path,
        name=dataset_name,
        input_seq_length=input_seq_length,
        extra_seq_length=n_rollout_steps,
        nl_backend=nl_backend,
    )
    data_test = H5Dataset(
        "test",
        dataset_path=dataset_path,
        name=dataset_name,
        input_seq_length=input_seq_length,
        extra_seq_length=n_rollout_steps,
        nl_backend=nl_backend,
    )

    return data_train, data_valid, data_test


def load_checkpoint(model_dir):
    """params_tree.pkl과 state_tree.pkl을 불러오는 유틸"""
    with open(os.path.join(model_dir, "params_tree.pkl"), "rb") as f:
        params = pickle.load(f)
    with open(os.path.join(model_dir, "state_tree.pkl"), "rb") as f:
        state = pickle.load(f)
    return params, state

def infer_one_step(cfg: Union[Dict, OmegaConf], state_input):
    """
    state_input: 모델이 기대하는 형식의 현재 상태 (positions, velocities 등)
    """
    # config 준비
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    check_cfg(cfg)

    # 데이터 메타데이터 준비
    data_train, _, _ = setup_data(cfg)
    metadata = data_train.metadata
    bounds = np.array(metadata["bounds"])
    box = bounds[:, 1] - bounds[:, 0]

    # case builder
    case = case_builder(
        box=box,
        metadata=metadata,
        input_seq_length=cfg.model.input_seq_length,
        cfg_neighbors=cfg.neighbors,
        cfg_model=cfg.model,
        noise_std=cfg.train.noise_std,
        external_force_fn=data_train.external_force_fn,
        dtype=cfg.dtype,
    )

    _, particle_type = data_train[0]

    # 모델 setup
    model_fn, MODEL = setup_model(
        cfg,
        metadata=metadata,
        homogeneous_particles=particle_type.max() == particle_type.min(),
        has_external_force=data_train.external_force_fn is not None,
        normalization_stats=case.normalization_stats,
    )
    model = hk.without_apply_rng(hk.transform_with_state(model_fn))

    # mixed precision
    policy = jmp.get_policy("params=float32,compute=float32,output=float32")
    hk.mixed_precision.set_policy(MODEL, policy)

    # checkpoint 로드
    params, state = load_checkpoint(cfg.load_ckp)

    # 1-step inference
    pred, new_state = model.apply(params, state, state_input)

    return pred, new_state

def setup_model(
    cfg,
    metadata: Dict,
    homogeneous_particles: bool = False,
    has_external_force: bool = False,
    normalization_stats: Optional[Dict] = None,
) -> Tuple[Callable, Type]:
    """Setup model based on cfg."""
    model_name = cfg.model.name.lower()
    input_seq_length = cfg.model.input_seq_length
    magnitude_features = cfg.model.magnitude_features

    if model_name == "gns":

        def model_fn(x):
            return models.GNS(
                particle_dimension=metadata["dim"],
                latent_size=cfg.model.latent_dim,
                blocks_per_step=cfg.model.num_mlp_layers,
                num_mp_steps=cfg.model.num_mp_steps,
                num_particle_types=NodeType.SIZE,
                particle_type_embedding_size=16,
            )(x)

        MODEL = models.GNS
    elif model_name == "segnn":
        # Hx1o vel, Hx0e vel, 2x1o boundary, 9x0e type
        node_feature_irreps = node_irreps(
            metadata,
            input_seq_length,
            has_external_force,
            magnitude_features,
            homogeneous_particles,
        )
        # 1o displacement, 0e distance
        edge_feature_irreps = Irreps("1x1o + 1x0e")

        def model_fn(x):
            return models.SEGNN(
                node_features_irreps=node_feature_irreps,
                edge_features_irreps=edge_feature_irreps,
                scalar_units=cfg.model.latent_dim,
                lmax_hidden=cfg.model.lmax_hidden,
                lmax_attributes=cfg.model.lmax_attributes,
                output_irreps=Irreps("1x1o"),
                num_mp_steps=cfg.model.num_mp_steps,
                n_vels=cfg.model.input_seq_length - 1,
                velocity_aggregate=cfg.model.velocity_aggregate,
                homogeneous_particles=homogeneous_particles,
                blocks_per_step=cfg.model.num_mlp_layers,
                norm=cfg.model.segnn_norm,
            )(x)

        MODEL = models.SEGNN
    elif model_name == "egnn":
        box = cfg.box
        if jnp.array(metadata["periodic_boundary_conditions"]).any():
            displacement_fn, shift_fn = space.periodic(jnp.array(box))
        else:
            displacement_fn, shift_fn = space.free()

        displacement_fn = jax.vmap(displacement_fn, in_axes=(0, 0))
        shift_fn = jax.vmap(shift_fn, in_axes=(0, 0))

        def model_fn(x):
            return models.EGNN(
                hidden_size=cfg.model.latent_dim,
                output_size=1,
                dt=metadata["dt"] * metadata["write_every"],
                displacement_fn=displacement_fn,
                shift_fn=shift_fn,
                normalization_stats=normalization_stats,
                num_mp_steps=cfg.model.num_mp_steps,
                n_vels=input_seq_length - 1,
                residual=True,
            )(x)

        MODEL = models.EGNN
    elif model_name == "painn":
        assert magnitude_features, "PaiNN requires magnitudes"
        radius = metadata["default_connectivity_radius"] * 1.5

        def model_fn(x):
            return models.PaiNN(
                hidden_size=cfg.model.latent_dim,
                output_size=1,
                n_vels=input_seq_length - 1,
                radial_basis_fn=models.painn.gaussian_rbf(20, radius, trainable=True),
                cutoff_fn=models.painn.cosine_cutoff(radius),
                num_mp_steps=cfg.model.num_mp_steps,
            )(x)

        MODEL = models.PaiNN
    elif model_name == "linear":

        def model_fn(x):
            return models.Linear(dim_out=metadata["dim"])(x)

        MODEL = models.Linear

    return model_fn, MODEL
