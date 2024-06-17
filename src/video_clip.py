import os
import subprocess

def get_all_video_files(directory):
    video_files = []
    for i in range(1, 19):
        video_files.append(directory + f"/{i}.mp4")
    return video_files

def remove_last_frame(video_path):
    # ffprobe -count_frames -show_entries stream=nb_read_frames  
    result = subprocess.run(
        ['ffprobe', '-count_frames', '-show_entries', 'stream=nb_read_frames', video_path],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )
    lines = result.stdout.split('\n')
    frame_count = 0
    for l in lines:
        if l.startswith("nb_read_frames"):
            frame_count = int(l.split("=")[1].strip())
            break;

    output_path = os.path.splitext(video_path)[0] + '_trimmed' + os.path.splitext(video_path)[1]
    output_path = "../data/videos/" + os.path.split(video_path)[1]
    subprocess.run(
        ['ffmpeg', '-y', '-i', video_path, '-vf', f'select=lte(n\,{frame_count-2})', '-vsync', 'vfr', output_path]
    )

    print(f"Processed video saved as: {output_path}")

def process_all_videos_in_directory(directory):
    video_files = get_all_video_files(directory)
    for video_file in video_files:
        print(f"Processing video: {video_file}")
        remove_last_frame(video_file)

directory = "../data/"
process_all_videos_in_directory(directory)
