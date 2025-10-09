# Dataset Processing Notebook

## 📘 Overview
This repository contains the `data_set_processing.ipynb` notebook, which is designed to preprocess and view raw data for downstream video generation tasks.

### 📂 Functions
- `print_unique_frame_counts()`: Simply prints unique numbers of frames.
- `get_video_ids_by_frame_count()`: Group Videos ID by number of thir frames.
- `is_similar()`: Function that returns true if img1 is similar to img2 using the Structural similarity index measure (SSIM).
- `show_video_with_pose()`: Show the video with the relative poses, optimized=True to optimize.
- `find_top_extreme_videos()`: Sort by length
- `print_unique_frame_dimensions()`: Print unique dimensions.

### 📊 Dataset Contents

The dataset contains the following fields for each sample:
- `frames`: RGB image sequences of shape (num_frames, width, height, 3)
- `x`, `y`: Normalized joint coordinates of shape (num_frames, 13)
- `visibility`: Visibility flags of each joint of shape (num_frames, 13)
- `pose`: A string representing the camera pose (e.g., 'front', 'back')
- `action`: A string label indicating one of the 15 predefined human actions
- `dimensions` : dimensions of the frames
- `nframes`  : number of frames

## 🧠 Purpose
The main goal of this notebook is to view a clean, optimized dataset that can be directly used for preparing training multimodal models. It ensures consistency and compatibility with frameworks that require structured input-output pairs such as:
- RGB frame sequences
- Human pose coordinates
- Action or caption labels
- Other custom modalities
