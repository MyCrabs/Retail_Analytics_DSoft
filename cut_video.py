from moviepy.editor import VideoFileClip

input = "input/after_lunch.mp4"
output = "input/after_lunch_1.mp4"

video = VideoFileClip(input)
duration = video.duration
print(f"Total time of video {duration/60:.2f} minutes {duration:.0f} seconds")

start_time = 6 * 60 - 5
end_time = 9 * 60 + 5

if end_time > duration:
    end_time = duration
elif start_time > end_time:
    raise ValueError("Unsuitable Time Setting")

final = video.subclip(start_time, end_time)
final.write_videofile(
    output,
    codec = 'libx264',
    preset = 'slow',
    bitrate = '4000k'
)
print(f"Done Save File in {output}")