import torch
from torchvision.transforms import v2
import torchvision.transforms.v2.functional as F
from torchvision import tv_tensors
from torch.utils.data import Dataset, DataLoader
from PIL import Image as Img
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pyproj import Transformer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def get_local_coordinates(coords, src_crs="4326", dst_crs="21781"):
    """
    Convert GPS coordinates to local coordinate reference system

    Args:
        coords (ndarray): GPS coordinates (in EPSG:4326 by default)
    """

    transformer = Transformer.from_crs(src_crs, dst_crs)
    xy_coords = transformer.transform(coords[:, 0], coords[:, 1])

    return np.array(xy_coords).T

def generate_training_three_tuples(dataset, lead_times=[0, 360], lead_time_interval=10, weighted=False, reference_times=[(-1, "D")]):
    """
    Generate 3-tuples comprising:
    - a conditioning image It_prev 24h (24 * 6 * 10 minutes) before the target time
    - a conditioning image I0 lead_time before the target time
    - the image at the target time
    """

    triples = []
    for lead in range(lead_times[0], lead_times[1] + lead_time_interval, lead_time_interval):
        intersection = dataset.data.index.intersection(dataset.data.index - pd.Timedelta(1440-lead, "min")).intersection(dataset.data.index - pd.Timedelta(1440, "min"))
        end = dataset.data.loc[dataset.data.loc[intersection].index + pd.Timedelta(1440, 'min')]["rowid"]
        start = dataset.data.loc[dataset.data.loc[intersection].index + pd.Timedelta(1440-lead, 'min')]["rowid"]
        prev = dataset.data.loc[intersection]["rowid"]
        lead_triples = np.concatenate([prev.values[..., np.newaxis], start.values[..., np.newaxis], end.values[..., np.newaxis]], axis=1)
        triples.append(lead_triples)
    
    idx_tuple = np.concatenate(triples)

    dst = np.zeros((len(dataset.data)))
    sort_idx = np.lexsort([idx_tuple[:, 2], idx_tuple[:, 1], idx_tuple[:, 0]])
        
    return idx_tuple[sort_idx], dst

class PanoramaNorthRoll(v2.Transform):
    """Roll 360 deg panorama to North heading"""

    def __init__(self, heading, angle=360):
        self.heading = heading
        self.angle = angle
        super().__init__()

    def _transform(self, inpt, params):

        _, image_height, image_width = F.get_dimensions(inpt)
        half_px = int(image_width / 2)
        original_width = int(image_width * self.angle / 360)
        heading_horiz_px = int(self.heading * original_width)

        if isinstance(inpt, tv_tensors.Image):
            img = torch.roll(inpt, half_px - heading_horiz_px, dims=-1)
            img = v2.functional.to_image(img)
        elif type(inpt) == torch.Tensor:
            img = torch.roll(inpt, half_px - heading_horiz_px, dims=-1)
        elif isinstance(inpt, Img.Image):
            img = F.to_tensor(inpt)
            img = torch.roll(img, half_px - heading_horiz_px, dims=-1)
            img = F.to_pil_image(img)

        return img

class PanoramaExpand(v2.Transform):
    """Expand panorama to 360 field of view"""

    def __init__(self, angle):
        self.angle = angle
        super().__init__()

    def _transform(self, inpt, params):

        _, image_height, image_width = F.get_dimensions(inpt)
        new_width = int(360 * image_width / self.angle)

        if self.angle != 360:
            new_shape = list(inpt.shape)
            new_shape[-1] = new_width
            img = torch.zeros(new_shape, dtype=inpt.dtype, device=inpt.device)
            if isinstance(inpt, tv_tensors.Image):
                img[..., :image_width] = inpt
            elif isinstance(inpt, Img.Image):
                img[..., :image_width] = F.to_tensor(inpt)
                img = F.to_pil_image(img)
        else:
            img = inpt

        if isinstance(inpt, tv_tensors.Image):
            return v2.functional.to_image(img)
        
        return img
    
class PanoramaContract(v2.Transform):
    """Undo PanoramaExpand — crops back to the original field of view"""

    def __init__(self, angle):
        self.angle = angle
        super().__init__()

    def _transform(self, inpt, params):

        _, image_height, image_width = F.get_dimensions(inpt)
        original_width = int(image_width * self.angle / 360)

        if self.angle != 360:
            return F.crop(inpt, 0, 0, image_height, original_width)
        else:
            return inpt


class PanoramaRandomCrop(v2.Transform):
    """Crop panorama at random location"""

    def __init__(self, size):
        super().__init__()

        if isinstance(size, int):
            size = [size, size]
        elif (isinstance(size, list) or isinstance(size, tuple)) and len(size) in {1, 2}:
            if len(size) == 1:
                size = size + size
            else:
                size = list(size)
        else:
            raise ValueError(
                f"size can be an integer, or a sequence of one or two integers, but got {size} instead."
            )
        self.size = size
        self.crop_height, self.crop_width = size

    def _get_params(self, flat_inputs):
        horiz_px = (torch.rand(size=(1,))).item()
        params = dict(horiz_px=horiz_px)
        return params

    def _transform(self, inpt, params):

        _, image_height, image_width = F.get_dimensions(inpt)
        crop_height, crop_width = self.crop_height, self.crop_width
        crop_horiz_px = int(params["horiz_px"] * (image_width - crop_width))

        return F.crop(inpt, 0, crop_horiz_px, crop_height, crop_width)
    
class Photocast2DDataset(Dataset):
    """
    Photocast dataset wrapper (panorama images and 2D weather fields)
    """

    def __init__(self,
                 csv_file,
                 image_root_dir,
                 img_transform=None):
        """
        Args:
            csv_file: Contains information on the timesteps available for training
            image_root_dir: root directory for panorama images
            weather_root_dir: root directory for 2D weather fields in netcdf format
            dem_file: an path to a netcdf file containing a DEM variable covering the area of interest
            weather_scaler_file: path to npz file containing mean and std for weather normalization
            viewshed_radius: radius in meters for cropping weather and DEM around camera location
            weighted_pairs: whether to weight training pairs based weighted sampling scheme
            w_transform: whether to apply normalization transform to weather data
            img_transform: torchvision transform to apply to images and corresponding masks
            lead_times: minimum and maximum lead times in minutes to include in training pairs
            reference_times: list of reference times for samples
        """
        self.data = pd.read_csv(csv_file, index_col=0)
        self.data.index = pd.to_datetime(self.data["reference"])

        # Netcdf file for 2024-12-29 broken TODO: exclude this somewhere else
        # self.data = self.data[self.data.index.date > pd.Timestamp("2025-01-01").date()]
        self.data["rowid"] = pd.RangeIndex(len(self.data))

        self.image_root_dir = image_root_dir

        self.img_transform = img_transform
        self.tf_expand = PanoramaExpand(angle=self.data["angle"].iloc[0])
        self.tf_roll = PanoramaNorthRoll(angle=self.data["angle"].iloc[0], heading=self.data["north"].iloc[0])

    def __len__(self):
        return len(self.data)
     
    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()

        # Load images: ref view, prev image(s), current image, target image
        I0 = F.to_image(Img.open(os.path.join(self.image_root_dir, self.data["path"].iloc[index])))

        # Expand panorama to 360 FOV
        _, H, W = F.get_dimensions(I0)
        I0 = self.tf_expand(I0)

        # Roll panorama to have North in center
        I0 = self.tf_roll(I0)

        if self.img_transform is not None:
            assert isinstance(I0, tv_tensors.Image)
            I0 = self.img_transform(I0)


        sample = {}
        sample["tensors"] = (I0,)
        sample["labels"] = 0 # dummy
        sample["view"] = self.data["cam"].iloc[0]

        return sample

def get_photocast_dataloader(cfg):
    """
    """

    transforms = v2.Compose([
        v2.Resize(cfg['data']['img_size'], interpolation=v2.InterpolationMode.BICUBIC),
        PanoramaRandomCrop(cfg['data']['img_size']),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    datasets = []

    for camera in tqdm(cfg["data"]["train"], "Loading training data"):  

        try:    
            datasets.append(
                Photocast2DDataset(csv_file=os.path.join(cfg['data']['image_root_dir'], "images", "2026-09-24_train_cams", f"{camera}.csv"),
                                    image_root_dir=cfg['data']["image_root_dir"],
                                    img_transform=transforms)
            )
        except IndexError as e:
            print(f"Error loading dataset for camera {camera}: {e}")
            continue

    dataset = torch.utils.data.ConcatDataset(datasets)

    return dataset
    
    # persistent_workers = True if cfg["data"]["num_workers"] > 0 else False
    # prefetch_factor = 4 if cfg["data"]["num_workers"] > 0 else None
    # dataloader = DataLoader(dataset, batch_size=cfg["data"]['batch_size'], shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True, persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)

    # return dataloader
