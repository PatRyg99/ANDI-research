import os
from typing import Union, List
from dataclasses import asdict

from sklearn.model_selection import train_test_split
import numpy as np
import andi

from configs.generate_data_config import GenerateDataConfig


def main() -> None:
    config = GenerateDataConfig()

    # Create andi dataset
    AD = andi.andi_datasets()
    dataset = AD.create_dataset(
        N=config.N,
        T=config.T,
        exponents=config.exponents,
        models=config.model,
        dimension=config.dimension,
    )

    # Split dataset
    train_data, test_data = train_test_split(
        dataset, train_size=config.train_size, stratify=dataset[:, 0]
    )
    val_data, train_data = train_test_split(
        test_data, train_size=0.5, stratify=test_data[:, 0]
    )

    # Extract trajectory and label
    X_train, y_train = (
        train_data[:, 2:].reshape(-1, config.T, config.dimension),
        train_data[:, 0],
    )
    X_val, y_val = (
        val_data[:, 2:].reshape(-1, config.T, config.dimension),
        val_data[:, 0],
    )
    X_test, y_test = (
        test_data[:, 2:].reshape(-1, config.T, config.dimension),
        test_data[:, 0],
    )

    # Save data
    np.save(os.path.join(config.out_path, "X_train.npy"), X_train)
    np.save(os.path.join(config.out_path, "y_train.npy"), y_train)
    np.save(os.path.join(config.out_path, "X_val.npy"), X_val)
    np.save(os.path.join(config.out_path, "y_val.npy"), y_val)
    np.save(os.path.join(config.out_path, "X_test.npy"), X_test)
    np.save(os.path.join(config.out_path, "y_test.npy"), y_test)


if __name__ == "__main__":
    main()
