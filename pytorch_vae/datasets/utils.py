from pathlib import Path


def get_paths_from_folders(images_dir):
    """Returns list of files in folders of input"""
    paths = []
    for folder in Path(images_dir).iterdir():
        if folder.suffix == ".txt":
            continue

        for p1 in folder.iterdir():
            if p1.suffix == ".JPEG":
                paths.append(p1)

            elif p1.is_dir():
                for p2 in p1.iterdir():
                    paths.append(p2)

    return paths