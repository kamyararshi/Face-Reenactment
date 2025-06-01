import face_alignment
import skimage.io
import numpy
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm
import os
import imageio
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def extract_bbox(frame, fa):
    if max(frame.shape[0], frame.shape[1]) > 640:
        scale_factor = max(frame.shape[0], frame.shape[1]) / 640.0
        frame = resize(frame, (int(frame.shape[0] / scale_factor), int(frame.shape[1] / scale_factor)))
        frame = img_as_ubyte(frame)
    else:
        scale_factor = 1
    frame = frame[..., :3]
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) == 0:
        return []
    return np.array(bboxes)[:, :-1] * scale_factor


def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def join(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB)


def process_video_and_save_frames(args, output_dir, video_id, video_name_without_ext):
    device = 'cpu' if args.cpu else 'cuda'
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=device)
    video_reader = imageio.get_reader(args.inp)

    # Create output directory for the video
    video_output_path = os.path.join(output_dir, video_id, video_name_without_ext)
    os.makedirs(video_output_path, exist_ok=True)

    fps = 30.0 # Default FPS if not obtainable from metadata

    trajectories = []
    
    # try:
    #     meta_data = video_reader.get_meta_data()
    #     if 'fps' in meta_data and meta_data['fps'] is not None:
    #         fps = float(meta_data['fps'])
    #     else:
    #         print(f"Warning: FPS metadata not found for {args.inp}. Defaulting to 30.0 FPS.")
    # except Exception as e:
    #     print(f"Error getting video metadata for {args.inp}: {e}. Defaulting to 30.0 FPS.")

    try:
        for i, frame in tqdm(enumerate(video_reader), desc=f"Processing {video_name_without_ext}"):
            frame_shape = frame.shape
            bboxes = extract_bbox(frame, fa)

            not_valid_trajectories = []
            valid_trajectories = []

            for trajectory in trajectories:
                tube_bbox = trajectory[0]
                intersection = 0
                for bbox in bboxes:
                    intersection = max(intersection, bb_intersection_over_union(tube_bbox, bbox))
                if intersection > args.iou_with_initial:
                    valid_trajectories.append(trajectory)
                else:
                    not_valid_trajectories.append(trajectory)
            
            # End current trajectories and process their frames (if any)
            for bbox, tube_bbox, start, end in not_valid_trajectories:
                if (end - start) > args.min_frames:
                    # Crop and save frames for these finished trajectories
                    # This requires re-reading the video for the specific segment
                    # A more efficient way would be to store frames or process all at once
                    # but for this structure, re-reading is simpler for now.
                    # For practical applications, consider storing frames or a single pass.
                    
                    # For simplicity and to fit the current structure, we'll extract the current frame's bbox
                    # and crop only the current frame `i` around the detected face.
                    if len(bboxes) > 0: # Ensure there's a detected face in the current frame
                        left, top, right, bot = tube_bbox
                        width = right - left
                        height = bot - top

                        width_increase = max(args.increase, ((1 + 2 * args.increase) * height - width) / (2 * width))
                        height_increase = max(args.increase, ((1 + 2 * args.increase) * width - height) / (2 * height))

                        left = int(left - width_increase * width)
                        top = int(top - height_increase * height)
                        right = int(right + width_increase * width)
                        bot = int(bot + height_increase * height)

                        top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
                        
                        # Crop the current frame
                        cropped_frame = frame[top:bot, left:right]
                        
                        # Resize to image_shape
                        if cropped_frame.shape[0] > 0 and cropped_frame.shape[1] > 0:
                            cropped_frame_resized = resize(cropped_frame, args.image_shape, anti_aliasing=True)
                            cropped_frame_ubyte = img_as_ubyte(cropped_frame_resized)
                            frame_filename = os.path.join(video_output_path, f"{i:05d}.png") # Save as PNG or JPG
                            imageio.imwrite(frame_filename, cropped_frame_ubyte)
                        else:
                            print(f"Skipping empty crop for {video_name_without_ext}, frame {i}")


            trajectories = valid_trajectories

            ## Assign bbox to trajectories, create new trajectories
            for bbox in bboxes:
                intersection = 0
                current_trajectory = None
                for trajectory in trajectories:
                    tube_bbox = trajectory[0]
                    current_intersection = bb_intersection_over_union(tube_bbox, bbox)
                    if intersection < current_intersection and current_intersection > args.iou_with_initial:
                        intersection = bb_intersection_over_union(tube_bbox, bbox)
                        current_trajectory = trajectory

                ## Create new trajectory
                if current_trajectory is None:
                    trajectories.append([bbox, bbox, i, i])
                else:
                    current_trajectory[3] = i
                    current_trajectory[1] = join(current_trajectory[1], bbox)

            # For active trajectories, crop and save the current frame 'i'
            for bbox, tube_bbox, start_idx, end_idx in trajectories:
                left, top, right, bot = tube_bbox
                width = right - left
                height = bot - top

                width_increase = max(args.increase, ((1 + 2 * args.increase) * height - width) / (2 * width))
                height_increase = max(args.increase, ((1 + 2 * args.increase) * width - height) / (2 * height))

                left = int(left - width_increase * width)
                top = int(top - height_increase * height)
                right = int(right + width_increase * width)
                bot = int(bot + height_increase * height)

                top, bot, left, right = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
                
                # Crop the current frame
                cropped_frame = frame[top:bot, left:right]
                
                # Resize to image_shape
                if cropped_frame.shape[0] > 0 and cropped_frame.shape[1] > 0:
                    cropped_frame_resized = resize(cropped_frame, args.image_shape, anti_aliasing=True)
                    cropped_frame_ubyte = img_as_ubyte(cropped_frame_resized)
                    frame_filename = os.path.join(video_output_path, f"{i:05d}.png") # Save as PNG or JPG
                    imageio.imwrite(frame_filename, cropped_frame_ubyte)
                else:
                    print(f"Skipping empty crop for {video_name_without_ext}, frame {i}")


    except Exception as e: # Catch broader exceptions during video processing
        print(f"Error processing {args.inp}: {e}")
    finally:
        video_reader.close() # Ensure video reader is closed

    # No need for commands here, as we are directly saving frames

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--dataset_path", required=True, help='Path to the root of the dataset (e.g., ./dataset)')
    parser.add_argument("--output_path", required=True, help='Path to the directory to save cropped frames')
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")
    parser.add_argument("--increase", default=0.1, type=float, help='Increase bbox by this amount')
    parser.add_argument("--iou_with_initial", type=float, default=0.25, help="The minimal allowed iou with inital bbox")
    parser.add_argument("--min_frames", type=int, default=1, help='Minimum number of frames for a trajectory to be processed (will be 1 for frame-by-frame saving)')
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")


    args = parser.parse_args()

    # Iterate through each ID folder
    for id_folder_name in sorted(os.listdir(args.dataset_path)):
        id_folder_path = os.path.join(args.dataset_path, id_folder_name)
        if os.path.isdir(id_folder_path) and id_folder_name.startswith('ID_'):
            print(f"Processing ID: {id_folder_name}")
            # Iterate through each video in the ID folder
            for video_filename in sorted(os.listdir(id_folder_path)):
                if video_filename.endswith('.mp4'): # or other video formats
                    video_path = os.path.join(id_folder_path, video_filename)
                    video_name_without_ext = os.path.splitext(video_filename)[0]

                    # Update args.inp for the current video
                    args.inp = video_path

                    print(f"  Processing video: {video_filename}")
                    process_video_and_save_frames(args, args.output_path, id_folder_name, video_name_without_ext)