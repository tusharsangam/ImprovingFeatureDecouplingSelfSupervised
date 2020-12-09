from PIL import Image
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torchvision.datasets import DatasetFolder
from torch.utils.data.dataloader import default_collate
import torch


#change default behavioir of ImageFolder Class to load data in the way favourable
class DatasetFolderCustom(DatasetFolder):
    def __init__(self, root: str,
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            unsupervised:bool = True,
            simCLR:bool=False
    ):
        super(DatasetFolderCustom, self).__init__(root, loader=loader, extensions=extensions,transform=transform,
                                            target_transform=target_transform )#is_valid_file=is_valid_file
        self.unsupervised = unsupervised
        self.simCLR = simCLR
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.unsupervised is True:
            if self.simCLR is True:
                path, _ = self.samples[index]
                sample = self.loader(path)
                if self.transform is not None:
                    samplei = self.transform(sample)
                    samplej = self.transform(sample)
                return samplei, samplej
            else:
                path, _ = self.samples[index]
                sample = self.loader(path)
                if self.transform is not None:
                    sample = self.transform(sample)
                #rotation labels
                target = torch.LongTensor([0,1,2,3])
                index = torch.LongTensor([index])
                return sample, target, index
        else:
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return sample, target



IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImageFolder(DatasetFolderCustom):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            unsupervised=True,
            simCLR=False
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file, unsupervised=unsupervised, simCLR=simCLR)
        self.imgs = self.samples

def custom_collate_fn(batch):
    batch = default_collate(batch)
    batch_size, rotations = batch[1].size()
    batch[1] = batch[1].view([batch_size*rotations])
    batch[2] = batch[2].squeeze_(dim=1)
    #batch[2] = batch[2].view([batch_size*rotations])
    return batch