import torch

class DummyVisionDataset:
    """
    Simulates loading batch data (visual tokens) for small-scale testing 
    without needing heavy dataset dependencies right away.
    """
    def __init__(self, num_samples=10, num_patches=196, embed_dim=1024):
        self.num_samples = num_samples
        self.num_patches = num_patches
        self.embed_dim = embed_dim

    def get_batch(self, batch_size=2):
        # Generates a random batch of visual patch tokens [B, N, D]
        return torch.randn(batch_size, self.num_patches, self.embed_dim)