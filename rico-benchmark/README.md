# D-RICO and EC-RICO Benchmarks

We provide annotation files for all tasks in the benchmarks. Each file contains a list of images, where each entry includes:

- **Image Path:** The image path.
- **Image Size:** The image size.
- **Absent Classes:** Classes not present in the entire task.
- **Task Info:** The task name and ID.
- **Bounding Box Labels:** Including category name and ID, bounding box coordinates, and bounding box mode.
- **Image ID:** The image ID.
- **Additional Information:** Some images may include extra details from the processing stage that are irrelevant to the benchmark.

---

## Using the Benchmark

Before using the benchmark, all datasets must be downloaded, and some require additional processing. The annotation files contain image paths that need to be updated to reflect the local storage location. A script is provided to automate this process (see below). Further details can be found in the paper and supplementary material.

---

## Setting the Correct Image Paths

A simple shell script is provided to update image paths. The script assumes that the downloaded dataset folder structure remains unchanged. However, since dataset structures may vary, manual corrections might be necessary.

### Configuring the Paths

1. **Update the `paths.sh` file:** 
   Configure the correct local paths for each dataset by specifying only the dataset’s main folder, not the specific image subfolders. 
   For example, for the **nuImages** dataset, set:
   ```sh
   NUIMAGES_PATH="your_path/nuImages"
   ```
   Even though an image might be located at your_path/nuImages/samples/CAM_FRONT/n01...7.jpg, only the main folder path should be defined in the script.

2. **Run the script:**
   After updating the file, execute the script to correct all paths.

4. **Verify the paths:**
   Use the check_paths.sh script to scan all annotation files and check whether the referenced image files exist. The script will return an error if a file is not found.