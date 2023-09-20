import torch
from torchvision import transforms
from torchvision import datasets

def do_dataloader(folder: str, batch_size=32, shuffle=True, tran = []):

    toTransform = [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
    for i in tran:
        toTransform.append(i)
    transform = transforms.Compose(toTransform)

    path = folder
    try:
        dataset = datasets.ImageFolder(root=path, transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    except Exception as e:
        error = Exception(f"Error: {__file__}: do_dataloader: {str(e)}")
        raise error

    return dataloader

#Questa funzione crea i dataloader per il train e il test
def make_dataloaders(datasets_path, batch_size=32, shuffle=True):

    train_textures_path = datasets_path + '/train/textures'
    train_masks_path = datasets_path + '/train/masks'
    test_textures_path = datasets_path + '/test/textures'
    test_masks_path = datasets_path + '/test/masks'

    try:
        dl_train_text = do_dataloader(train_textures_path, batch_size=batch_size, shuffle=shuffle)
        dl_train_mask = do_dataloader(train_masks_path, batch_size=batch_size, shuffle=shuffle, tran = [transforms.Grayscale(num_output_channels=1)])
        dl_test_text = do_dataloader(test_textures_path, batch_size=batch_size, shuffle=shuffle)
        dl_test_mask = do_dataloader(test_masks_path, batch_size=batch_size, shuffle=shuffle, tran = [transforms.Grayscale(num_output_channels=1)])
    except Exception as e:
        print(e)
        raise e
    return dl_train_text, dl_train_mask, dl_test_text, dl_test_mask
