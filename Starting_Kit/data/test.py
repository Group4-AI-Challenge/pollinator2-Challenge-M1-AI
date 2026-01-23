import h5py

filename = 'file.h5'

def explore_h5(name, obj):
    print(f"Path: {name}")
    if isinstance(obj, h5py.Dataset):
        print(f"  - Type: Dataset")
        print(f"  - Shape: {obj.shape}")
        print(f"  - Data Type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"  - Type: Group")
    
    if obj.attrs:
        print(f"  - Attributes: {dict(obj.attrs)}")
    print("-" * 30)

with h5py.File(filename, 'r') as f:
    f.visititems(explore_h5)