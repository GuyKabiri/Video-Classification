import torch

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    alpha = 4

    def __init__(self):
        super().__init__()

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // PackPathway.alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list