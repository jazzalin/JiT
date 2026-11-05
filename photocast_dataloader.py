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
import xarray as xr
from tqdm import tqdm
import os

def normalize_dem(dem, dem_min=0, dem_max=4545):
    """Apply global normalization to DEM"""

    norm_dem = (dem - dem_min) / (dem_max - dem_min)
    return norm_dem

def normalize_weather(w, w_stats):
    """Apply global normalization to weather, appending a NaN mask channel"""

    N, H, W = w.shape
    w_stats = torch.Tensor(w_stats)
    norm_w = w.reshape(N, -1)
    norm_w = (norm_w - w_stats[:, [0]]) / w_stats[:, [1]]
    norm_w = norm_w.reshape(N, H, W)

    nan_mask = torch.isnan(norm_w).any(axis=0)
    norm_w = norm_w.nan_to_num(0)
    norm_w = torch.concat([norm_w, nan_mask[None]], axis=0)

    return norm_w

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
                 img_transform=None,
                 label=0):
        """
        Args:
            csv_file: Contains information on the timesteps available for training
            image_root_dir: root directory for panorama images
            label: class id returned with every sample (camera index for class-conditional training)
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
        self.label = label

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
        sample["labels"] = self.label
        sample["view"] = self.data["cam"].iloc[0]

        return sample

class Photocast2DDatasetZarr(Dataset):
    """
    Photocast dataset wrapper (panorama images and 2D weather fields read from a shared zarr store).
    Each sample is an (It_prev, I0, It) triple with It_prev 24h before It and I0 lead_time before It.
    Adapted from photocast.dataloader.dataset.Photocast2DDatasetZarr without the holistic images.
    """

    def __init__(self,
                 csv_file,
                 image_root_dir,
                 shared_weather_ds,
                 weather_scaler_file,
                 viewshed_radius=4000,
                 w_transform=True,
                 img_transform=None,
                 lead_times=[0, 360],
                 reference_times=[(-1, "D")],
                 date_ranges=None):
        """
        Args:
            csv_file: Contains information on the timesteps available for training
            image_root_dir: root directory for panorama images
            shared_weather_ds: xarray.Dataset with the spatial weather fields for the entire study domain
            weather_scaler_file: path to npz file containing mean and std for weather normalization
            viewshed_radius: radius in meters for cropping weather and DEM around camera location
            w_transform: whether to apply normalization transform to weather data
            img_transform: torchvision transform to apply to images and corresponding masks
            lead_times: minimum and maximum lead times in minutes to include in training triples
            reference_times: list of reference times for samples
            date_ranges: list of [start, end] datetimes (inclusive) of the images to keep, None keeps all
        """
        self.data = pd.read_csv(csv_file, index_col=0)
        self.data.index = pd.to_datetime(self.data["reference"])

        if date_ranges is not None:
            keep = np.zeros(len(self.data), dtype=bool)
            for start, end in date_ranges:
                keep |= (self.data.index >= pd.Timestamp(start)) & (self.data.index <= pd.Timestamp(end))
            self.data = self.data[keep]
        self.data["rowid"] = pd.RangeIndex(len(self.data))

        self.image_root_dir = image_root_dir

        # Weather crop settings
        self.viewshed_radius = viewshed_radius
        self.bbox = self.get_bbox(
            (self.data["latitude"].iloc[0], self.data["longitude"].iloc[0]),
            viewshed_radius
        )

        # weather columns to load from zarr dataset
        self.weather_columns = ['FF', 'PT', 'WG', 'CT', 'RS', 'Z0', 'TW', 'TD', 'TG', 'ZS',
                                'RR', 'SU', 'ZM', 'DD', 'TT']
        self._weather_ds = shared_weather_ds[self.weather_columns]

        # load scaler file for weather normalization
        weather_std_file = np.load(weather_scaler_file)
        weather_std = []
        for c in self.weather_columns:
            if c in weather_std_file:
                weather_std.append(weather_std_file[c])

        weather_std.append([0, 366]) # scaling for doy
        weather_std.append([0, 24 * 60]) # scaling for tod in minutes

        self.std_scaler = np.stack(weather_std, axis=0)

        self.img_transform = img_transform
        self.tf_expand = PanoramaExpand(angle=self.data["angle"].iloc[0])
        self.tf_roll = PanoramaNorthRoll(angle=self.data["angle"].iloc[0], heading=self.data["north"].iloc[0])
        self.w_transform = w_transform

        # Generate index
        self.pairs, self.dst = generate_training_three_tuples(self, lead_times=lead_times, reference_times=reference_times)

    def __len__(self):
        return len(self.pairs)

    def get_bbox(self, gps_coords, radius):
        """Get bounding box approximating camera viewshed

        Args:
            gps_coords (tuple): GPS latitude-longitude pair
            radius (float): viewshed radius
        """

        coords = get_local_coordinates(np.array(gps_coords)[None])
        bbox = [coords[:, 0].item() - radius,
                coords[:, 1].item() - radius,
                coords[:, 0].item() + radius,
                coords[:, 1].item() + radius]
        return bbox

    def get_dem_crop(self, dem):
        """Get DEM crop based on viewshed radius

        Args:
            dem (xarray.Dataset): dem dataset containing 'DEM' variable
        """

        dem = dem.sel(
            chx=slice(self.bbox[0], self.bbox[2]),
            chy=slice(self.bbox[1], self.bbox[3])
        ).DEM.values

        # Invert y-axis to have N pointing up
        dem = torch.flip(torch.Tensor(dem), dims=(0,))

        return dem

    def get_weather_crop(self, dt):
        """Get weather crop based on viewshed radius

        Args:
            dt (pd.Timestamp): datetime of the image
        """

        wt = self._weather_ds.sel(
            datetime=dt.strftime(format="%Y-%m-%dT%H:%M:%S"),
            x=slice(self.bbox[0], self.bbox[2]),
            y=slice(self.bbox[1], self.bbox[3]),
            ).to_array().values

        # Add time to weather vector
        doy = dt.dayofyear
        minutes = (dt - pd.to_datetime(dt.date())).total_seconds() / 60
        time = np.array([doy, minutes])
        time = np.broadcast_to(time[:, np.newaxis, np.newaxis], (2, wt.shape[1], wt.shape[2]))

        wt = np.concatenate([wt, time])

        # Invert y-axis to have N pointing up
        wt = torch.flip(torch.Tensor(wt), dims=(1,))

        return wt

    def __getitem__(self, index):

        if torch.is_tensor(index):
            index = index.tolist()

        It_prev_idx, I0_idx, It_idx = self.pairs[index]

        # Load images: prev image (24h before target), current image, target image
        It_prev = F.to_image(Img.open(os.path.join(self.image_root_dir, self.data["path"].iloc[It_prev_idx])))
        I0 = F.to_image(Img.open(os.path.join(self.image_root_dir, self.data["path"].iloc[I0_idx])))
        It = F.to_image(Img.open(os.path.join(self.image_root_dir, self.data["path"].iloc[It_idx])))

        # Crop corresponding 2D weather maps
        wt_prev = self.get_weather_crop(self.data.index[It_prev_idx])
        w0 = self.get_weather_crop(self.data.index[I0_idx])
        wt = self.get_weather_crop(self.data.index[It_idx])

        # Expand panorama to 360 FOV, keeping track of the expanded region with a mask image
        _, H, W = F.get_dimensions(I0)
        view_mask = F.to_image(torch.ones((1, H, W)))
        I0, It, It_prev, view_mask = self.tf_expand(I0, It, It_prev, view_mask)

        # Roll panorama to have North in center
        I0, It, It_prev, view_mask = self.tf_roll(I0, It, It_prev, view_mask)

        # Encode the heading as cos(theta) and sin(theta) frames matching image dimensions,
        # so the mapping between panorama and weather heading can be learned
        _, H, W = F.get_dimensions(I0)
        cos = torch.cos(torch.linspace(0, 2 * torch.pi, W).unsqueeze(0).expand(H, -1))
        sin = torch.sin(torch.linspace(0, 2 * torch.pi, W).unsqueeze(0).expand(H, -1))
        angle_mask = torch.stack([cos, sin, torch.zeros_like(sin)], dim=0)
        angle_mask = (angle_mask + 1.0) / 2.0
        # Assuming North centered images, roll angle mask to have 0 at half-width
        angle_mask = F.to_image(torch.roll(angle_mask, int(W / 2)))

        if self.img_transform is not None:
            assert all(isinstance(I, tv_tensors.Image) for I in (I0, It, It_prev, view_mask, angle_mask))
            I0, It, It_prev, view_mask, angle_mask = self.img_transform(I0, It, It_prev, view_mask, angle_mask)
            # Drop redundant dimensions
            view_mask = F.rgb_to_grayscale(view_mask)

        if self.w_transform:
            w0 = normalize_weather(w0, self.std_scaler)
            wt = normalize_weather(wt, self.std_scaler)
            wt_prev = normalize_weather(wt_prev, self.std_scaler)

        sample = {}
        sample["tensors"] = (I0, It, It_prev, w0, wt, wt_prev, angle_mask, view_mask)
        sample["view"] = self.data["cam"].iloc[I0_idx]
        sample["datetimes"] = [dt.strftime('%Y-%m-%d_%H:%M') for dt in self.data.index[[It_prev_idx, I0_idx, It_idx]]]

        return sample

def get_photocast_transforms(cfg, crop="center"):
    """
    Image transforms shared by training and evaluation (output in [-1, 1]).
    crop: "center" for a fixed crop at the panorama center, i.e. the North-facing view after PanoramaNorthRoll,
    "random" for a crop at a random heading (same crop for all frames and masks of a sample),
    None to keep the full panorama (resized to the crop height)
    """

    crops = {
        "center": v2.CenterCrop(cfg['data']['img_size']),
        "random": PanoramaRandomCrop(cfg['data']['img_size']),
        None: v2.Identity(),
    }
    return v2.Compose([
        v2.Resize(cfg['data']['img_size'][0], interpolation=v2.InterpolationMode.BICUBIC),
        crops[crop],
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def camera_csv_path(cfg, camera):
    return os.path.join(cfg['data']['image_root_dir'], "images", "2026-09-24_train_cams", f"{camera}.csv")

def get_photocast_dataloader(cfg):
    """
    Training dataset over all cameras in cfg["data"]["train"], with the weather zarr store opened once
    and shared by every camera. Samples are dicts with
    "tensors": (I0, It, It_prev, w0, wt, wt_prev, angle_mask, view_mask).
    """

    transforms = get_photocast_transforms(cfg, crop="random")

    shared_weather = xr.open_zarr(os.path.join(cfg["data"]["weather_root_dir"], "inca.zarr"), consolidated=True, chunks=None)

    datasets = []
    cameras = []

    for camera in tqdm(cfg["data"]["train"], "Loading training data"):

        try:
            datasets.append(
                Photocast2DDatasetZarr(csv_file=camera_csv_path(cfg, camera),
                                       image_root_dir=cfg['data']["image_root_dir"],
                                       shared_weather_ds=shared_weather,
                                       weather_scaler_file=cfg["data"]["weather_scaler_file"],
                                       viewshed_radius=cfg["data"]["viewshed_radius"],
                                       img_transform=transforms,
                                       lead_times=cfg["data"]["lead_times"],
                                       date_ranges=cfg["data"].get("date_ranges"))
            )
            cameras.append(camera)
        except IndexError as e:
            print(f"Error loading dataset for camera {camera}: {e}")
            continue

    dataset = torch.utils.data.ConcatDataset(datasets)
    dataset.cameras = cameras
    print(f"Loaded {len(cameras)}/{len(cfg['data']['train'])} cameras, {len(dataset)} samples")
    for camera, d in zip(cameras, datasets):
        print(f"  {camera}  {len(d)} samples ({len(d.data)} images)")

    return dataset
