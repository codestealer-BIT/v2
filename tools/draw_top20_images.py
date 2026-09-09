import pandas as pd
from PIL import Image
from data_pipeline_utils import TiktokDataFusedPipeline
import base64
import io
import os
import argparse

def draw_csv_grid_with_captions_safe(csv_path, frame_width=224, video_gap=50, rows_per_block=32, output_dir=None):
    """
    Draw CSV as a grid of images with first column separated as identifier,
    sorting by first column, saving captions, and skipping rows with errors.

    Args:
        csv_path (str): Path to CSV.
        get_video_info_func (callable): Function returning dict with keys 'frame' (PIL.Image) and 'caption' (str).
        frame_width (int): Width to resize each frame to.
        gap (int): Gap in pixels after first column.
        rows_per_block (int): How many rows per vertical block.
        output_caption_csv (str): Filepath to save the captions CSV.

    Returns:
        PIL.Image: Final concatenated image.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = df.sort_values(by='item_id', ignore_index=True)

    data_pipeline = TiktokDataFusedPipeline()

    n_rows, n_cols = df.shape
    row_images = []

    for idx, row in df.iterrows():
        try:
            print(f"Processing row {idx}")

            video_blocks = []
            captions_row = []

            count = 0
            for col_idx, item_id in enumerate(row):
                if count > 5:
                    break
                count += 1
                info = data_pipeline.request_all_info(item_id)
                video_frames = info.get('video', []) if info is not None else []
                video_frames = [base64.b64decode(video_frames[i]) for i in range(len(video_frames))]
                video_frames = [Image.open(io.BytesIO(image_str)).convert('RGB') for image_str in video_frames]

                frames_to_draw = []

                if len(video_frames) == 0:
                    # Draw 3 empty frames
                    empty_frame = Image.new('RGB', (frame_width,int(frame_width * 16 / 9)), (100, 100, 100))
                    frames_to_draw = [empty_frame] * 3
                else:
                    n = len(video_frames)
                    indices = [0, n//2, n-1] if n > 2 else list(range(n))
                    for i in indices:
                        img = video_frames[i]
                        h = int(img.height * frame_width / img.width)
                        img = img.resize((frame_width,h))
                        frames_to_draw.append(img)

                # Concatenate frames horizontally for this video
                total_width = sum(f.width for f in frames_to_draw)
                max_height = max(f.height for f in frames_to_draw)
                video_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))
                x_offset = 0
                for f in frames_to_draw:
                    video_img.paste(f, (x_offset, 0))
                    x_offset += f.width

                video_blocks.append(video_img)

                # Add video gap except for last video
                if col_idx == 0:  # first column: small gap after for identifier
                    gap_img = Image.new('RGB', (video_gap, max_height), (255, 255, 255))
                    video_blocks.append(gap_img)
                elif col_idx < len(row)-1:
                    gap_img = Image.new('RGB', (video_gap, max_height), (255, 255, 255))
                    video_blocks.append(gap_img)

            # Concatenate all videos horizontally
            total_width = sum(v.width for v in video_blocks)
            max_height = max(v.height for v in video_blocks)
            row_img = Image.new('RGB', (total_width, max_height), (255, 255, 255))
            x_offset = 0
            for v in video_blocks:
                row_img.paste(v, (x_offset, 0))
                x_offset += v.width

            # Save row image
            # filename = f"row_{idx}.png"
            # filepath = os.path.join(output_dir, filename)
            # row_img.save(filepath)

            row_images.append(row_img)

            # Once we have two rows, stack and save
            if len(row_images) == 2:
                total_width = max(r.width for r in row_images)
                total_height = sum(r.height for r in row_images)
                stacked_img = Image.new('RGB', (total_width, total_height), (255,255,255))
                y_offset = 0
                for i, r in enumerate(row_images):
                    stacked_img.paste(r, (0, y_offset))
                    y_offset += r.height

                    if i == 0:
                        y_offset += video_gap

                filename = f"rows_{idx-1}_{idx}.png"
                filepath = os.path.join(output_dir, filename)
                stacked_img.save(filepath)

                # Clear row_images for next batch
                row_images = []

        except Exception as e:
            print(f"Skipping row {idx} due to error: {e}")
            continue

    

if __name__ == "__main__":
    # Example usage:
    parser = argparse.ArgumentParser(description="Draw grid of videos with captions")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the output grid images")
    args = parser.parse_args()

    draw_csv_grid_with_captions_safe(args.csv_path,
                                     output_dir=args.output_dir)
    
