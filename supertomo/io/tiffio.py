from .tiffile import TiffFile


def get_tiff(filename):
    tags = {}
    with TiffFile(filename) as image:
        # Get images
        images = image.asarray()
        # Get tags
        page = image[0]
        for tag in page.tags.values():
            tags[tag.name] = tag.value
    return images, tags
