import subprocess

ffmpeg = 'ffmpeg'

def Options():

    user_input_dict = {}
    
    user_input_dict["input_file"] = "./generatedvideo.avi"
    user_input_dict["output_file"] = "./generatedvideo.mp4"
    user_input_dict["video_codec"] = "libx264"
    user_input_dict["audio_codec"] = "aac" 
    user_input_dict["audio_bitrate"] = "196k"
    user_input_dict["sample_rate"] = "44100"
    user_input_dict["encoding_speed"] = "fast"
    user_input_dict["crf"] = "22"
    user_input_dict["frame_size"] = "1280x720"

    return user_input_dict

def buildFFmpegCommand(input_file,output_file):

    final_user_input = Options()
    final_user_input["input_file"] = input_file
    final_user_input["output_file"] = output_file

    commands_list = [
        ffmpeg,
        "-i",
        final_user_input["input_file"],
        "-c:v",
        final_user_input["video_codec"],
        "-preset",
        final_user_input["encoding_speed"],
        "-crf",
        final_user_input["crf"],
        "-s",
        final_user_input["frame_size"],
        "-c:a",
        final_user_input["audio_codec"],
        "-b:a",
        final_user_input["audio_bitrate"],
        "-ar",
        final_user_input["sample_rate"],
        "-pix_fmt",
        "yuv420p",
        final_user_input["output_file"]
        ]

    return commands_list

def runFFmpeg(commands):

    if subprocess.run(commands).returncode == 0:
        print ("FFmpeg Script Ran Successfully")
    else:
        print ("There was an error running your FFmpeg script")

