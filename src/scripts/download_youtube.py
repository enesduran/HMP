import os 
import glob
from pytube import YouTube



def download_youtube_clip(url, download_folder):
    return YouTube(url).streams.get_highest_resolution().download(output_path=download_folder)


download_folder = "./data/rgb_data/in_the_wild"
 


# if you want to add a video, add to the last part. 

flip_list = ["guitar.mp4", "guitar practice.mp4"]

video_weblink_dict = {
                      "https://www.youtube.com/watch?v=_8aoIizn68E": "Making HANDMADE Derby Shoes in Embossed Cordovan Leather.mp4",
                      "https://www.youtube.com/watch?v=-5b9adbHmZo": "tesbih.mp4",
                      "https://www.youtube.com/watch?v=IeYZPuAI_xk": "Forging a Copper Damascus Katana.mp4",
                      "https://www.youtube.com/watch?v=CsHEiXxsjmQ": "Card At ANY Number - Impossible Card Trick PerformanceTutorial.mp4",
                      "https://www.youtube.com/watch?v=j5AcnZSwduY": "guitar.mp4",
                      "https://www.youtube.com/watch?v=O3H9M39bHsU": "chess notation.mp4",
                      "https://www.youtube.com/watch?v=UeSEjyYzSpE": "guitar practice.mp4",
                      "https://www.youtube.com/shorts/3FvJ-nHwcMM": "one-handed-card-trick.mp4",
                      "https://www.youtube.com/shorts/3g1SaKt3cx0": "watermelon.mp4", 
                      "https://www.youtube.com/watch?v=UVKnVPRklCc": "baby sign language.mp4",
                      "https://www.youtube.com/watch?v=6_gXiBe9y9A": "ASL alphabet.mp4",
                      "https://www.youtube.com/watch?v=fPyCcoaRMS8": "1 Handed Signing.mp4",
                      "https://www.youtube.com/shorts/ed4CAAXWUyU": "drum.mp4",
                      "https://www.youtube.com/watch?v=79FBoliI1ag&t=91s": "one hand shuffle.mp4",
                      "https://www.youtube.com/watch?v=TlfbxprcyMo": "one hand shuffle1.mp4",
                      "https://www.youtube.com/watch?v=JH2zD6lhH_4&t=2s": "one hand shuffle2.mp4",
                      "https://www.youtube.com/watch?v=0CA0EXlxumE&t=1s": "baklava.mp4",
                        }
               
               
video_parts_dict = {
                    "https://www.youtube.com/watch?v=_8aoIizn68E": [["00:00:15.750 ", "00:00:19"], ["00:15:36.500", "00:15:40"], ["00:21:51.300", "00:21:55"], ["00:30:35.750", "00:30:40"]],   # cand 0-3
                    "https://www.youtube.com/watch?v=-5b9adbHmZo": [["00:00:02", "00:00:05"], ["00:00:35", "00:00:55"]],     # cand 4-5                                                                   
                    "https://www.youtube.com/watch?v=IeYZPuAI_xk": [["00:08:59", "00:09:15"],],
                    "https://www.youtube.com/watch?v=CsHEiXxsjmQ": [["00:02:51", "00:03:15"]],
                    "https://www.youtube.com/watch?v=j5AcnZSwduY": [["00:02:55", "00:03:15"]],          # cand 8
                    "https://www.youtube.com/watch?v=O3H9M39bHsU": [["00:01:28", "00:01:41"], ["00:01:45.750", "00:01:55"]],  
                    "https://www.youtube.com/watch?v=UeSEjyYzSpE": [["00:00:00", "00:00:26"], ["00:06:30", "00:06:53"], ["00:07:10", "00:07:35"]], # cand 11-13
                    "https://www.youtube.com/shorts/3FvJ-nHwcMM": [["00:00:01.500", "00:00:20.500"]],
                    "https://www.youtube.com/shorts/3g1SaKt3cx0": [["00:00:05", "00:00:11"], ["00:00:05", "00:00:35"]],    # cand 15-16 
                    "https://www.youtube.com/watch?v=UVKnVPRklCc": [["00:02:08.500", "00:02:19"], ["00:03:12", "00:03:19"], ["00:03:28", "00:03:34"], ["00:04:04.200", "00:04:19.500"]],   # cand 17-20
                    "https://www.youtube.com/watch?v=6_gXiBe9y9A": [["00:00:20", "00:01:23.200"]],          # cand 21
                    "https://www.youtube.com/watch?v=fPyCcoaRMS8": [["00:00:04.500", "00:00:15"], ["00:01:37", "00:01:44"]],   # cand 22-23
                    "https://www.youtube.com/shorts/ed4CAAXWUyU": [["00:00:02.300", "00:00:07"], ["00:00:08", "00:00:13"], ["00:00:17.900", "00:00:25"], ["00:00:51.300", "00:00:55.000"]],   # cand 24-27
                    "https://www.youtube.com/watch?v=79FBoliI1ag&t=91s": [["00:00:50", "00:01:02"]],
                    "https://www.youtube.com/watch?v=TlfbxprcyMo": [["00:01:37", "00:01:54"], ["00:02:16", "00:02:34"], ["00:05:20", "00:05:47"]],   # cand 29-31
                    "https://www.youtube.com/watch?v=JH2zD6lhH_4&t=2s": [["00:00:05", "00:00:35"]],  # cand 32
                    "https://www.youtube.com/watch?v=0CA0EXlxumE&t=1s": [["00:02:52", "00:02:59"], ["00:08:56", "00:09:02"]],  # cand 33-34
                    }

cand_idx = 0

for i, (vid_link, vidname) in enumerate(video_weblink_dict.items()):
    
    vid_file_path = f"{download_folder}/{vidname}"
    
    # first download the video if not exists 
    if not os.path.exists(vid_file_path):
        download_youtube_clip(vid_link, download_folder)
    
    trim_parts = video_parts_dict[vid_link]
    
    for trim_part in trim_parts:
        
        output_folder = f"{download_folder}/cand_{cand_idx}"

        os.makedirs(output_folder, exist_ok=True)
        
        trimmed_vid_file_path = f"{output_folder}/rgb_raw.mp4"


        # then trim the video 
        if not os.path.isfile(trimmed_vid_file_path):
             
            hflip = vidname in flip_list
                        
            if hflip:
                command_str = f"/usr/bin/ffmpeg -i '{vid_file_path}' -vf hflip -ss {trim_part[0]} -to {trim_part[1]}  -r 30 {output_folder}/rgb_raw.mp4 -y"
            else:
                command_str = f"/usr/bin/ffmpeg -i '{vid_file_path}' -ss {trim_part[0]} -to {trim_part[1]} -c:v libx264 -r 30 {output_folder}/rgb_raw.mp4 -y"
                # command_str = f"ffmpeg -i '{vid_file_path}' -ss {trim_part[0]} -to {trim_part[1]} -c:v /usr/NX/lib/libx264.so  -r 30 {output_folder}/rgb_raw.mp4 -y"

        

            os.system(command_str)    
            
        # then extract the frames
        if not os.path.exists(f"{output_folder}/rgb"): 
            os.makedirs(f"{output_folder}/rgb", exist_ok=True)
            # make them video as well
            command_str = f"/usr/bin/ffmpeg -i {output_folder}/rgb_raw.mp4 -r 30 '{output_folder}/rgb/%06d.jpg'"
            os.system(command_str)    
            
 
        cand_idx += 1
    
    
    
    
