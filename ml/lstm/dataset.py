"""Dataset PyTorch pour le LSTM de direction."""


def build_loaders(X_train, y_train, X_val, y_val, batch_size: int = 64):
    """
    Crée les DataLoaders train et val depuis des arrays numpy.

    Import lazy de torch pour ne pas le charger inutilement.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    def _to_loader(X, y, shuffle: bool):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return _to_loader(X_train, y_train, shuffle=True), \
           _to_loader(X_val,   y_val,   shuffle=False)
