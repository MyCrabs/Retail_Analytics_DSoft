from moviepy.editor import VideoFileClip

input_path = "input/cam1.mp4"
output_path = "input/cam1_cut.mp4"

video = VideoFileClip(input_path)
duration = video.duration
print(f"⏱️ Tổng thời lượng: {duration/60:.2f} phút ({duration:.0f} giây)")

# Cắt từ 6 phút 50s đến 8 phút 50s
start_time = 6 * 60 + 50   # 410 giây
end_time   = 8 * 60 + 50   # 530 giây

# Kiểm tra
if end_time > duration:
    end_time = duration
if start_time >= end_time:
    raise ValueError("Khoảng thời gian cắt không hợp lệ!")

final_clip = video.subclip(start_time, end_time)

final_clip.write_videofile(
    output_path,
    codec="libx264",
    audio_codec="aac",
    preset="medium",
    bitrate="3000k"
)

print(f"🎬 Đã cắt xong! File lưu tại: {output_path}")
