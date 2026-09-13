from typing import Callable


def build_hdf5_datamodule(
    batch_size: int,
    num_workers: int,
):
    from dataset.DataReader import HDF5DataModule

    return HDF5DataModule(
        "/data2/StormWave/SEVIR",
        batch_size=batch_size,
        num_workers=num_workers,
    )


def build_sevir_datamodule(
    batch_size: int,
    num_workers: int,
    seed: int,
):
    from dataset.SEVIR import SEVIRLightningDataModule

    return SEVIRLightningDataModule(
        seq_len=24,
        batch_size=batch_size,
        dataset_name="sevir",
        num_workers=num_workers,
        start_date=(2017, 5, 1),
        end_date=(2019, 12, 31),
        train_test_split_date=(2019, 6, 1),
        sevir_dir="/data3/SEVIR",
        seed=seed,
        aug_mode="1",
    )


DATASET_REGISTRY: dict[str, Callable[..., object]] = {
    "hdf5": build_hdf5_datamodule,
    "sevir": build_sevir_datamodule,
}


def build_datamodule(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    seed: int,
):
    if dataset_name == "sevir":
        return build_sevir_datamodule(
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )

    return DATASET_REGISTRY[dataset_name](
        batch_size=batch_size,
        num_workers=num_workers,
    )
