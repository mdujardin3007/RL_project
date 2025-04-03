# display_video.py

import os
import base64
from IPython.display import HTML, display

def show_video():
    video_folder = './video'
    video_files = os.listdir(video_folder)
    if video_files:
        video_path = os.path.join(video_folder, video_files[0])
        with open(video_path, 'rb') as f:
            mp4 = f.read()
        data_url = "data:video/mp4;base64," + base64.b64encode(mp4).decode()
        html = f"""
        <video width=700 controls>
            <source src="{data_url}" type="video/mp4">
        </video>
        """
        display(HTML(html))
    else:
        print("No video file found in", video_folder)

if __name__ == "__main__":
    show_video()
