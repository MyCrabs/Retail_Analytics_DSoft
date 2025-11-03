from moviepy.editor import VideoFileClip

input = "input/Dsoft_after.mp4"
output = "input/after_lunch.mp4"

video = VideoFileClip(input)
duration = video.duration
print(f"Tổng thời lượng video: {duration/60:.2f} phut ({duration:.0f} giay)")

start_time = 6 * 60 + 50
end_time = 8 * 60 + 50

if end_time > duration:
    end_time = duration
if start_time >= end_time:
    raise ValueError('Khoang thoi gian cat ko phu hop')

final_clip = video.subclip(start_time, end_time)
final_clip.write_videofile(
    output,
    codec = "libx264",
    preset = 'medium',
    bitrate = '3000k'
)
print(f"Da cat xong file va luu tai {output}")