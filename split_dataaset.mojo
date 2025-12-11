from python import Python


fn main() raises:
    # Import Python modules
    var shutil = Python.import_module("shutil")
    var datetime = Python.import_module("datetime")
    var pathlib = Python.import_module("pathlib")
    var random = Python.import_module("random")
    var time = Python.import_module("time")

    # Set random seed
    random.seed(42)

    # Paths
    var Path = pathlib.Path
    var SOURCE_DIR = Path("data")
    var DEST_DIR = Path("dataset")

    # Split ratios
    var TRAIN_RATIO = 0.70
    var VAL_RATIO = 0.15
    var TEST_RATIO = 0.15

    print("="* 80)
    print("DATASET SPLITTING TOOL (Mojo)")
    print("=" * 80)
    print("Started:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("Source:", SOURCE_DIR)
    print("Destination:", DEST_DIR)
    print(
        "Split ratio: Train="
        + String(TRAIN_RATIO * 100)
        + "%, Val="
        + String(VAL_RATIO * 100)
        + "%, Test="
        + String(TEST_RATIO * 100)
        + "%"
    )
    print("=" * 80)

    # Start timer
    var start_time = time.time()

    # Create destination directories
    var splits = Python.list()
    splits.append("train")
    splits.append("val")
    splits.append("test")

    for i in range(3):
        var split = splits[i]
        var split_dir = DEST_DIR / split

        if split_dir.exists():
            print("\nWARNING:", split_dir, "already exists!")
            print("Deleting and recreating...")
            shutil.rmtree(split_dir)
            print("Deleted", split_dir)

        _ = split_dir.mkdir(
            parents=Python.evaluate("True"), exist_ok=Python.evaluate("True")
        )

    # Get all class folders
    var class_folders = Python.list()
    for item in SOURCE_DIR.iterdir():
        if item.is_dir():
            class_folders.append(item)

    # Sort class folders
    class_folders.sort()

    print("\nFound", len(class_folders), "classes")
    print("=" * 80)

    var py_int = Python.import_module("builtins").int
    var total_images = py_int(0)
    var total_train = py_int(0)
    var total_val = py_int(0)
    var total_test = py_int(0)

    # Process each class
    for class_folder in class_folders:
        var class_name = class_folder.name

        # Get all image files
        var image_files = Python.list()
        var extensions = Python.list()
        extensions.append("*.jpg")
        extensions.append("*.jpeg")
        extensions.append("*.png")
        extensions.append("*.JPG")
        extensions.append("*.JPEG")
        extensions.append("*.PNG")

        for ext in extensions:
            for img in class_folder.glob(ext):
                image_files.append(img)

        if len(image_files) == 0:
            print("WARNING: No images found in", class_name)
            continue

        # Shuffle images
        random.shuffle(image_files)

        # Calculate split indices
        var n_images = len(image_files)
        var n_train = py_int(n_images * TRAIN_RATIO)
        var n_val = py_int(n_images * VAL_RATIO)
        var n_test = n_images - n_train - n_val

        # Split images using Python list slicing
        var py_slice = Python.import_module("builtins").slice
        var train_images = image_files.__getitem__(py_slice(0, n_train))
        var val_images = image_files.__getitem__(
            py_slice(n_train, n_train + n_val)
        )
        var test_images = image_files.__getitem__(
            py_slice(n_train + n_val, n_images)
        )

        # Create class directories in each split
        for split in splits:
            var class_dir = DEST_DIR / split / class_name
            _ = class_dir.mkdir(
                parents=Python.evaluate("True"),
                exist_ok=Python.evaluate("True"),
            )

        # Copy images to respective directories
        for img in train_images:
            shutil.copy2(img, DEST_DIR / "train" / class_name / img.name)

        for img in val_images:
            shutil.copy2(img, DEST_DIR / "val" / class_name / img.name)

        for img in test_images:
            shutil.copy2(img, DEST_DIR / "test" / class_name / img.name)

        # Update counters
        total_images = total_images.__add__(n_images)
        total_train = total_train.__add__(n_train)
        total_val = total_val.__add__(n_val)
        total_test = total_test.__add__(n_test)

        print(
            class_name + ":",
            n_images,
            "total -> Train:",
            n_train,
            "Val:",
            n_val,
            "Test:",
            n_test,
        )

    # Calculate elapsed time
    var elapsed_time = time.time() - start_time

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("Total images processed:", total_images)
    var train_pct = py_int(total_train) / py_int(total_images) * 100
    var val_pct = py_int(total_val) / py_int(total_images) * 100
    var test_pct = py_int(total_test) / py_int(total_images) * 100
    print("Train:", total_train, "(", train_pct, "%)")
    print("Val:", total_val, "(", val_pct, "%)")
    print("Test:", total_test, "(", test_pct, "%)")
    print("=" * 80)
    print("\n✅ Dataset split completed successfully!")
    print("\nDirectory structure:")
    print("  ", DEST_DIR)
    print("    train/", total_train, "images")
    print("    val/", total_val, "images")
    print("    test/", total_test, "images")
    print()
    print("⏱️  Total time:", elapsed_time, "seconds")
    var minutes = py_int(elapsed_time) / 60
    print("⏱️  Total time:", minutes, "minutes")
    var speed = py_int(total_images) / elapsed_time
    print("📊 Processing speed:", speed, "images/second")
    print("🏁 Completed:", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 80)
