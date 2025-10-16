from moviepy.editor import VideoFileClip

# === Nhập đường dẫn video ===
input_path = "input/cam1.mp4"      # thay bằng tên thật của bạn
output_path = "input/cam1_2.mp4"

# === Mở video ===
video = VideoFileClip(input_path)

# === Lấy tổng thời lượng video ===
duration = video.duration  # tính bằng giây
print(f"⏱️ Tổng thời lượng: {duration/60:.2f} phút ({duration:.0f} giây)")

# === Xác định khoảng cần cắt ===
start_time = 10 * 60      # phút 9 -> 540 giây
end_time = duration      # đến hết video

# Kiểm tra nếu video ngắn hơn
if start_time >= duration:
    raise ValueError("Video ngắn hơn 9 phút, không thể cắt đoạn này!")

# === Cắt đoạn video cần giữ ===
final_clip = video.subclip(start_time, end_time)

# === Xuất video ===
final_clip.write_videofile(
    output_path,
    codec="libx264",       # nén H.264 phổ biến
    audio_codec="aac",     # giữ âm thanh
    preset="medium",       # tốc độ xử lý
    bitrate="3000k"        # chất lượng hợp lý
)

print(f"✅ Đã cắt xong! File lưu tại: {output_path}")
