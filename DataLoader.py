import os
import cv2
import glob
import queue
import numpy as np

class DatasetVideoCapture:
    def __init__(self, streamer, val, folder_index, virtual_video_path=None):
        self.streamer = streamer
        self.val = val
        self.folder_index = folder_index
        self.virtual_video_path = virtual_video_path  
        # Store the file list here
        self.scene_name, self.frame_filenames = streamer.get_scene_info(val, folder_index)

        self.current_frame_idx = 0
        self.total_frames = len(self.frame_filenames)
        
        self.width = 0
        self.height = 0
        if self.total_frames > 0:
            sample = streamer.get_frame(val, folder_index, 0)
            self.height, self.width, _ = sample.shape
    
        self.cached_frames =  queue.Queue()
        self.is_done = False
        self.starvation = False
        self.starvation_threshold = 5 

        print('loading frames to buffer...', end='', flush=True)
        for i in range(20): # Preload some frames
            self.load_to_cache()
            if i%10 == 0:
                print('.', end='', flush=True) 
        print('\n\n')

    def read(self):
        if not self.cached_frames.empty():
            
            if self.cached_frames.qsize() < self.starvation_threshold and not self.is_done:
                self.starvation = True
                
            return self.cached_frames.get()
        else: # reached the end of the video
            return False, None

    def load_to_cache(self, ignore_errors=False):
        

        if self.current_frame_idx >= self.total_frames: # No more frames to read
            self.is_done = True
            self.starvation = False
            return
    

        try:
            # Use the efficient get_frame method we defined earlier
            frame = self.streamer.get_frame(self.val, self.folder_index, self.current_frame_idx)
            self.current_frame_idx += 1
            self.cached_frames.put((True,frame))
            
            if self.cached_frames.qsize() >= self.starvation_threshold * 2:
                self.starvation = False
            
        except Exception as e:
            if not ignore_errors:
                print(f"\n[ArgoverseCap] Error reading frame {self.current_frame_idx}: {e}")
                self.cached_frames.put((False, None))

    def get(self, propId):
        """
        Mimics cv2.VideoCapture.get(propId).
        Handles:
            3: CV_CAP_PROP_FRAME_WIDTH
            4: CV_CAP_PROP_FRAME_HEIGHT
            7: CV_CAP_PROP_FRAME_COUNT (optional bonus)
        """
        if propId == 3: 
            return self.width
        if propId == 4: 
            return self.height
        if propId == 7:
            return self.total_frames
        return 0

    def release(self):
        """
        Mimics cv2.VideoCapture.release().
        Nothing to close for file-based streaming, but required for API compatibility.
        """
        pass

class DatasetStreamer:
    def __init__(self, is_MOT = False):
        """
        Initializes the dataset structure.
        
        Args:
            root_dir (str): Path to the 'images' folder containing train/val/test splits.
        """
        self.root_dir = './Datasets/MOT17/images' if is_MOT else './Datasets/Argoverse/Argoverse-1.1/images'
        # Structure: {'train': [ [list_of_frame_paths_scene_1], [list_of_frame_paths_scene_2] ... ], 'val': ...}
        self.data_map = {
            'train': [],
            'val': []
        }
        
        print(f"Scanning dataset at: {self.root_dir}...")
        
        # Populate the data map
        for split in ['train', 'val']:
            split_path = os.path.join(self.root_dir, split)
            
            if not os.path.exists(split_path):
                print(f"Warning: {split} folder not found at {split_path}")
                continue
            
            # Get all Scene folders (UUIDs) and sort them to maintain consistent indexing
            # We use os.scandir for better performance than os.listdir
            scene_folders = sorted([f.path for f in os.scandir(split_path) if f.is_dir()])
            
            for scene_path in scene_folders:
                # Target specific subfolder as per prompt
                camera_dir = os.path.join(scene_path, "img1" if is_MOT else "ring_front_center")
                
                if os.path.exists(camera_dir):
                    # Get all .jpg frames and sort them so frame_index 0 -> t=0
                    frames = sorted(glob.glob(os.path.join(camera_dir, "*.jpg")))
                    if frames:
                        self.data_map[split].append(frames)
                        
        # Statistics
        self.train_count = len(self.data_map['train'])
        self.val_count = len(self.data_map['val'])
        
        total_train_frames = sum(len(scene) for scene in self.data_map['train'])
        total_val_frames = sum(len(scene) for scene in self.data_map['val'])

        # Delete first 60% of data_maps scenes duration <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< BE CAREFUL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        for split in ['train', 'val']:
            for scene_idx in range(len(self.data_map[split])):
                total_frames = len(self.data_map[split][scene_idx])
                cutoff = int(total_frames * 0.6)
                # cutoff = 0 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< BE CAREFUL >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                self.data_map[split][scene_idx] = self.data_map[split][scene_idx][cutoff:]
            print(f"Trimmed 60% of scenes in {split} split.")


        print("--- Initialization Complete ---")
        print(f"Train Scenes: {self.train_count} | Total Frames: {total_train_frames}")
        print(f"Val Scenes:   {self.val_count} | Total Frames: {total_val_frames}")

    def get_frame(self, val=True, folder_index=0, frame_index=0):
        """
        Retrieves a specific frame efficiently.

        Args:
            val (bool): If True, look in 'val' split, else 'train'.
            folder_index (int): Index of the scene (UUID folder).
            frame_index (int): Index of the image frame within that scene.

        Returns:
            numpy.ndarray: The image loaded via OpenCV (BGR format).
        """
        split_key = 'val' if val else 'train'
        scenes = self.data_map[split_key]
        
        # Error handling for indices
        if folder_index >= len(scenes) or folder_index < 0:
            raise IndexError(f"Folder index {folder_index} out of range for {split_key} (Max: {len(scenes)-1})")
        
        scene_frames = scenes[folder_index]
        
        if frame_index >= len(scene_frames) or frame_index < 0:
            raise IndexError(f"Frame index {frame_index} out of range for scene {folder_index} (Max: {len(scene_frames)-1})")
        
        # Retrieve path and load image
        image_path = scene_frames[frame_index]
        
        # cv2.imread is fast and standard for computer vision
        frame = cv2.imread(image_path)
        
        if frame is None:
            raise IOError(f"Failed to load image at {image_path}")
            
        return frame
    
    def get_frame_GT_label(self, val=True, folder_index=0, frame_index=0):
        """
        Retrieves a specific frame efficiently.

        Args:
            val (bool): If True, look in 'val' split, else 'train'.
            folder_index (int): Index of the scene (UUID folder).
            frame_index (int): Index of the image frame within that scene.

        Returns:
            np.ndarray: The loaded labels in YOLO format.
        """
        def load_yolo_file(filepath):
            """Loads YOLO file: (class x y w h)"""
            if filepath is None or not os.path.exists(filepath):
                return np.empty((0, 5)) 
            try:
                data = []
                with open(filepath, 'r') as f:
                    for line in f:
                        parts = list(map(float, line.strip().split()))
                        if len(parts) >= 5:
                            data.append(parts)
                if not data:
                    return np.empty((0, 5)) 
                return np.array(data, dtype=np.float32)
            except Exception:
                return np.empty((0, 5)) 

        split_key = 'val' if val else 'train'
        scenes = self.data_map[split_key]
        
        # Error handling for indices
        if folder_index >= len(scenes) or folder_index < 0:
            raise IndexError(f"Folder index {folder_index} out of range for {split_key} (Max: {len(scenes)-1})")
        
        scene_frames = scenes[folder_index]
        
        if frame_index >= len(scene_frames) or frame_index < 0:
            raise IndexError(f"Frame index {frame_index} out of range for scene {folder_index} (Max: {len(scene_frames)-1})")
        
        # Retrieve path and load image
        image_path = scene_frames[frame_index]
        
        label_path = image_path.replace("/images/", "/labels/").replace(".jpg", ".txt")
        
        labels = load_yolo_file(label_path)
        
        # concate confidence=1 for GT labels to match the shape of detection labels (class x y w h conf)
        if labels.shape[1] == 5:  # If GT labels don't have confidence, add a confidence column with value 1
            conf_column = np.ones((labels.shape[0], 1), dtype=np.float32)
            labels = np.hstack((labels, conf_column))  # Now shape is (N, 6)

        # put class in the last column for GT labels to match the shape of detection labels (x y w h conf class)
        labels = np.hstack((labels[:, 1:5], labels[:, 5:6], labels[:, 0:1]))  # Now shape is (N, 6) with class at the end

        # convert (xc yc w h conf class) to (x1 y1 x2 y2 conf class)
        labels[:, 0] = labels[:, 0] - labels[:, 2] / 2  # x1 = xc - w/2
        labels[:, 1] = labels[:, 1] - labels[:, 3] / 2  # y1 = yc - h/2
        labels[:, 2] = labels[:, 0] + labels[:, 2]  # x2 = x1 + w
        labels[:, 3] = labels[:, 1] + labels[:, 3]  # y2 = y1 + h

        return labels

    def get_scene_length(self, val=True, folder_index=0):
        """Helper to get the number of frames in a specific scene."""
        split_key = 'val' if val else 'train'
        if 0 <= folder_index < len(self.data_map[split_key]):
            return len(self.data_map[split_key][folder_index])
        return 0
    
    def cap(self, val=True, folder_index=0, generate_output_video = True):
        """
        Factory method that returns a cv2.VideoCapture-like object
        for the specified scene.
        """
        # Generate the mirror path for RT_detections
        virtual_video_path = self.create_detection_path(val, folder_index, generate_output_video = generate_output_video)

        return DatasetVideoCapture(self, val, folder_index, virtual_video_path)

    def get_scene_name(self, val=True, folder_index=0):
        """
        Returns the specific scene name (UUID folder name) for a given index.
        
        Args:
            val (bool): If True, look in 'val' split, else 'train'.
            folder_index (int): Index of the scene.
            
        Returns:
            str: The name of the folder (e.g., '00c561b9-2057...').
        """
        split_key = 'val' if val else 'train'
        scenes = self.data_map[split_key]

        if folder_index >= len(scenes) or folder_index < 0:
            raise IndexError(f"Folder index {folder_index} out of range for {split_key}")

        # Retrieve the first file path in that scene
        first_frame_path = scenes[folder_index][0]
        
        # Structure is: .../root/split/SCENE_UUID/ring_front_center/frame.jpg
        # We need to go up 2 directories from the file to get the scene folder
        scene_dir = os.path.dirname(os.path.dirname(first_frame_path))
        
        return os.path.basename(scene_dir)
    
    def get_scene_info(self, val=True, folder_index=0):
        """
        Returns the scene name and a list of all frame filenames for a given index
        by parsing the paths stored in self.data_map.

        Args:
            val (bool): If True, look in 'val' split, else 'train'.
            folder_index (int): Index of the scene.

        Returns:
            tuple: (scene_name, frame_filenames)
                   - scene_name (str): The name of the folder (e.g., '00c561b9...').
                   - frame_filenames (list): List of strings (e.g., ['ring_front_center_3159...jpg', ...]).
        """
        split_key = 'val' if val else 'train'
        scenes = self.data_map[split_key]

        # Safety Check
        if folder_index >= len(scenes) or folder_index < 0:
            raise IndexError(f"Folder index {folder_index} out of range for {split_key} (Max: {len(scenes)-1})")

        # Get the list of full file paths for the specific scene
        # Example: ['./Datasets/.../UUID/ring_front_center/img1.jpg', './Datasets/.../UUID/ring_front_center/img2.jpg']
        full_paths = scenes[folder_index]

        # 1. Extract Scene Name (UUID)
        # We take the first file path and go up two levels:
        # File: .../Argoverse/Argoverse-1.1/images/val/<UUID>/ring_front_center/<file.jpg>
        # dirname(file) -> .../ring_front_center
        # dirname(dirname(file)) -> .../<UUID>
        first_file_path = full_paths[0]
        scene_dir_path = os.path.dirname(os.path.dirname(first_file_path))
        scene_name = os.path.basename(scene_dir_path)

        # 2. Extract Frame Filenames
        # Convert full paths to just filenames (e.g. 'img1.jpg') using os.path.basename
        frame_filenames = [os.path.basename(p) for p in full_paths]

        return scene_name, frame_filenames

    def create_detection_path(self, val, folder_index, generate_output_video = True):
        """
        Creates the RT_detections directory tree and returns a virtual video path.
        """
        split_key = 'val' if val else 'train'
        # Example: ./Datasets/Argoverse/Argoverse-1.1/images/val/UUID/ring_front_center
        real_scene_path = os.path.dirname(self.data_map[split_key][folder_index][0])
    

        # Swap 'images' for 'RT_detections' in the path
        detection_dir = real_scene_path.replace("/images", "/RT_detections")

        # Create the directory if it doesn't exist
        # if not os.path.exists(detection_dir) and generate_output_video:
        #     os.makedirs(detection_dir, exist_ok=True)
        #     print(f"Created detection directory: {detection_dir}")

        return detection_dir

# --- Example Usage ---
if __name__ == "__main__":
    # Initialize the loader
    loader = DatasetStreamer(is_MOT = True)

    # Example: Get 5th frame from the 1st folder in Validation set
    try:
        frame = loader.get_frame(val=False, folder_index=0, frame_index=4)
        print(f"Successfully loaded frame with shape: {frame.shape}")
        
        # Optional: Show frame if running locally (comment out if on a headless server)
        cv2.imshow("Stream Test", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except IndexError as e:
        print(e)
