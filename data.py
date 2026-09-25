import h5py
import numpy as np
from pathlib import Path
from loguru import logger
from natsort import natsorted
import torch

from malvar_he_cutler import demosaic_interp, demosaic_malvar

class FastSpadLoader:
    """High-speed data loader optimized for interactive UI scrubbing of SPAD cubes."""
    
    def __init__(self, file_path: str | Path, rotation: int = 0, demosaic: bool = True):
        self.file_path = Path(file_path)
        self.is_h5 = self.file_path.suffix.lower() == ".h5"
        
        # Validate and store rotation configuration
        if rotation not in [0,90,180,270]:
            raise ValueError(f"Rotation must be 0, 90, 180, or 270 degrees. Got {rotation}")
        self.rotation = rotation
        # Calculate how many 90-degree counter-clockwise rotations are needed
        self.k_rot = {0: 0, 90: 1, 180: 2, 270: 3}[rotation]
        
        self.height = 0
        self.width = 0
        self.total_frames = 0
        self.do_demosaic = demosaic

        if self.is_h5:
            self._init_h5()
        else:
            self._init_npy()

        logger.info(f"Loaded SPAD cube: {self.width}x{self.height} (Rotated: {self.rotation}°) with {self.total_frames} scrubbable frames.")

    def _init_h5(self):
        self.h5_file = h5py.File(self.file_path, "r")
        self.group_path = self._find_data_group_path(self.h5_file)
        if not self.group_path:
            raise ValueError(f"Could not find a valid numeric data group in {self.file_path}")
            
        self.group = self.h5_file[self.group_path]
        all_keys = natsorted(self.group.keys())
        self.dataset_keys = [k for k in all_keys if k.isdigit() or ('frame' in k.lower())]
        
        if not self.dataset_keys:
            if isinstance(self.group, h5py.Dataset):
                shape = self.group.shape
                if len(shape) == 3:
                    self.total_frames, h, w = shape
                    self.is_3d_dataset = True
                    # Swap dimensions if dimensions alter spatial presentation
                    self.height, self.width = (w, h) if self.k_rot in [1, 3] else (h, w)
                    return
            raise ValueError(f"H5 file contains no sequential frame datasets.")
            
        self.is_3d_dataset = False
        self.total_frames = len(self.dataset_keys)
        
        # Read the first frame's original spatial dimensions
        first_frame = self.group[self.dataset_keys[0]][...].squeeze()
        h, w = first_frame.shape[-2], first_frame.shape[-1]
        # Dynamically invert layout parameters if rotated 90 or 270 degrees
        self.height, self.width = (w, h) if self.k_rot in [1, 3] else (h, w)

    def _init_npy(self):
        self.memmap_cube = np.load(self.file_path, mmap_mode="r")
        shape = self.memmap_cube.shape
        
        if len(shape) == 3:
            self.total_frames, h, w_packed = shape
            w = w_packed * 8
            self.height, self.width = (w, h) if self.k_rot in [1, 3] else (h, w)
        else:
            raise ValueError(f"Unsupported NPY shape: {shape}. Expected 3D array.")

    def _find_data_group_path(self, h5_obj, current_path="") -> str | None:
        if isinstance(h5_obj, h5py.Dataset):
            return current_path
        for key in h5_obj.keys():
            sub_obj = h5_obj[key]
            path = f"{current_path}/{key}" if current_path else key
            if isinstance(sub_obj, h5py.Group):
                child_keys = list(sub_obj.keys())
                if child_keys and (child_keys[0].isdigit() or 'frame' in child_keys[0].lower()):
                    return path
                deep_path = self._find_data_group_path(sub_obj, path)
                if deep_path:
                    return deep_path
        return None


    def _apply_rotation(self, frame: np.ndarray) -> np.ndarray:
        """Applies high-speed orientation matrix transformation using memory views."""
        if self.k_rot > 0:
            return np.rot90(frame, k=self.k_rot, axes=(0, 1)).copy()
        return frame


    def get_frame(self, index: int) -> np.ndarray:
        """Fetches a single 2D slice from disk instantly with orientation adjustment."""
        if index < 0 or index >= self.total_frames:
            raise IndexError(f"Frame index {index} out of bounds.")
            
        if self.is_h5:
            if self.is_3d_dataset:
                frame = self.group[index, :, :].astype(np.float32)
            else:
                key = self.dataset_keys[index]
                frame = self.group[key][...].squeeze().astype(np.float32)
        else:
            frame = self.memmap_cube[index, :, :]
            if frame.dtype == np.uint8 and self.width > frame.shape[1]:
                frame = np.unpackbits(frame, axis=1).astype(np.float32)

        frame = self._apply_rotation(frame.astype(np.float32))
        if self.do_demosaic:
            frame = demosaic_malvar(torch.tensor(frame), bayer_pattern='bggr').cpu().numpy()
        return frame

    def get_averaged_frame(self, start_idx: int, window_size: int) -> np.ndarray:
        """Computes a fast temporal average window over a range of frames."""
        end_idx = min(start_idx + window_size, self.total_frames)
        
        if window_size == 1:
            return self.get_frame(start_idx)
            
        # Collect frames in slice range
        frames = [self.get_frame(i) for i in range(start_idx, end_idx)]
        return np.mean(np.stack(frames, axis=0), axis=0)

    def __len__(self):
        return self.total_frames



    # @njit(parallel=True)
    # def fast_spad_average(self, cube, start, end):
    #     # Fast multi-threaded C loop over array
    #     return np.mean(cube[:, :, start:end], axis=2)
