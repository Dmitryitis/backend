def image_processor(file_name, variations, storage=None, replace=True):
    from api.tasks.images import process_photo_image

    process_photo_image.delay(file_name, variations, replace=replace)
    return False  # prevent default rendering