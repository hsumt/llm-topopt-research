import imageio


def save_gif(frame_paths: list, out_path: str, fps: int = 5):
    """
    Function stitching the images together

    Parameters
    ----------
    frame_paths : list of PNG paths (in order)
    out_path    : output GIF path
    fps         : frames per second
    """
    frames = [imageio.imread(p) for p in frame_paths]
    imageio.mimsave(out_path, frames, fps=fps)