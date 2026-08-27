from moviepy import VideoFileClip  # Updated for modern MoviePy
import os

def split_video_into_4_parts(input_filepath):
    video = VideoFileClip(input_filepath)
    total_duration = video.duration
    
    part_duration = total_duration / 4.0
    filename, ext = os.path.splitext(input_filepath)
    
    print(f"Total duration: {total_duration:.2f}s. Splitting into 4 parts of {part_duration:.2f}s each...")

    for i in range(4):
        start_time = i * part_duration
        end_time = (i + 1) * part_duration
        
        # --- FIXED LINE FOR MOVIEPY v2.0+ ---
        part_clip = video.subclipped(start_time, end_time) 
        
        output_filepath = f"{filename}_part{i+1}{ext}"
        print(f"Exporting {output_filepath} (From {start_time:.2f}s to {end_time:.2f}s)...")
        
        part_clip.write_videofile(output_filepath, codec="libx264", audio_codec="aac")
        
    video.close()
    print("Video successfully split!")

split_video_into_4_parts("Railway_video.mp4")