# ch name : GStreamer Video Transcoding Command

import json
import os

def generate_gst_command(command_type, **kwargs):
    if command_type == "file_conversion":
        return f"gst-launch-1.0 filesrc location={kwargs['input_file']} ! qtdemux ! queue ! h264parse ! qtic2vdec ! qtic2venc ! queue ! h265parse ! mp4mux ! filesink location={kwargs['output_file']}"
    elif command_type == "camera_output":
        return f"gst-launch-1.0 -e qtiqmmfsrc camera={kwargs['camera']} name=camsrc ! video/x-raw, format={kwargs['format']}, width={kwargs['width']}, height={kwargs['height']}, framerate={kwargs['framerate']} ! waylandsink x={kwargs['x']} y={kwargs['y']} width={kwargs['sink_width']} height={kwargs['sink_height']} sync=false"
    elif command_type == "pipeline_app":
        return f"gst-pipeline-app -e qtiqmmfsrc name=camsrc ! video/x-raw, format={kwargs['format']}, width={kwargs['width']}, height={kwargs['height']}, framerate={kwargs['framerate']} ! multifilesink enable-last-sample=false location={kwargs['yuv_location']} max-files=5 camsrc.image_1 ! \"image/jpeg, width={kwargs['width']}, height={kwargs['height']}, framerate={kwargs['framerate']}\" ! multifilesink location={kwargs['jpg_location']} sync=true async=false"
    elif command_type == "pipeline_app_yuv_jpeg":
        return (f'gst-pipeline-app -e qtiqmmfsrc name=camsrc ! video/x-raw, format={kwargs["format"]}, '
                f'width={kwargs["width"]}, height={kwargs["height"]}, framerate={kwargs["framerate"]}! '
                f'multifilesink enable-last-sample=false location={kwargs["yuv_location"]} max-files=5 '
                f'camsrc.image_11 "image/jpeg,width={kwargs["width"]},height={kwargs["height"]}, '
                f'framerate={kwargs["framerate"]}" ! multifilesink location={kwargs["jpg_location"]} sync=true async=false')

    elif command_type == "camera_wayland_encode":
        return (f'gst-launch-1.0 -e qtiqmmfsrc camera={kwargs["camera"]} name=camsrc video/x-raw, format={kwargs["format"]}, '
                f'width={kwargs["width"]}, height={kwargs["height"]}, framerate={kwargs["framerate1"]}! waylandsink x={kwargs["x"]} y={kwargs["y"]} '
                f'width={kwargs["sink_width"]} height={kwargs["sink_height"]} sync=false camsrc. ! video/x-raw, format={kwargs["format"]},width={kwargs["width"]}, '
                f'height={kwargs["height"]}, framerate={kwargs["framerate2"]} ! queue ! qtic2venc control-rate=constant target-bitrate={kwargs["bitrate"]}! '
                f'queue ! h264parse ! mp4mux ! queue ! filesink location={kwargs["output_file"]}')
    
    elif command_type == "pipeline_app_encode_jpeg":
        return (f'gst-pipeline-app -e qtiqmmfsrc camera={kwargs["camera"]} name=camsrc ! video/x-raw, format={kwargs["format"]}, '
                f'width={kwargs["width"]}, height={kwargs["height"]}, framerate={kwargs["framerate"]} ! queue ! qtic2venc ! '
                f'queue ! h264parse ! mp4mux ! queue ! filesink location={kwargs["video_location"]} camsrc.image_1 "image/jpeg, '
                f'width={kwargs["width"]}, height={kwargs["height"]}" ! multifilesink location={kwargs["jpg_location"]} sync=true async=false')
    
    elif command_type == "dual_camera_multifilesink":
        return (f'gst-launch-1.0 -e qtiqmmfsrc camera={kwargs["camera1"]} name=camsrc video/x-raw, format={kwargs["format1"]}, '
                f'width={kwargs["width1"]}, height={kwargs["height1"]}, framerate={kwargs["framerate1"]}! multifilesink enable-last-sample=false location={kwargs["location1"]} max-files-5 '
                f'qtiqmmfsrc camera={kwargs["camera2"]} ! video/x-raw, format={kwargs["format2"]}, width={kwargs["width2"]}, height={kwargs["height2"]}, '
                f'framerate={kwargs["framerate2"]} ! multifilesink enable-last-sample=false location={kwargs["location2"]} max-files-5')

    elif command_type == "pipeline_app_wayland_jpeg":
        return (f'gst-pipeline-app -e qtiqmmfsrc name=camsrc camera={kwargs["camera"]} video/x-raw, format={kwargs["format"]}, '
                f'width={kwargs["width"]}, height={kwargs["height"]} ! queue waylandsink fullscreen={kwargs["fullscreen"]} async={kwargs["is_async"]} '
                f'camsrc.image_1 ! "image/jpeg,width={kwargs["width"]}, height={kwargs["height"]}" ! multifilesink location={kwargs["jpg_location"]} sync=true async=false')

    elif command_type == "pipeline_app_video_image":
        commands = []
        for w, h in zip(kwargs["widths"], kwargs["heights"]):
            command = (f'gst-pipeline-app -e qtiqmmfsrc name=camsrc camera={kwargs["camera"]} ! video/x-raw, format={kwargs["format"]}, '
                       f'width={w}, height={h} queue ! qtic2venc ! queue! h264parse! mp4mux ! queue ! '
                       f'filesink location="{kwargs["mp4_location"]}" camsrc.image_1 ! "image/jpeg,width={kwargs["img_width"]},height={kwargs["img_height"]}" ! '
                       f'multifilesink location={kwargs["jpg_location"]} sync=true async=false')
            commands.append(command)
        return commands

    else:
        return None

def generate_command_id(command_type, **kwargs):
    if command_type == "file_conversion":
        input_name = kwargs['input_file'].split("/")[-1].split('.')[0]
        output_name = kwargs['output_file'].split("/")[-1].split('.')[0]
        return f"{input_name}_to_{output_name}_h264_to_h265"
    elif command_type == "camera_output":
        return f"camera{kwargs['camera']}_{kwargs['width']}x{kwargs['height']}_{kwargs['framerate']}fps_to_waylandsink"
    elif command_type == "pipeline_app":
        return f"{kwargs['format']}_{kwargs['width']}x{kwargs['height']}_{kwargs['framerate']}fps_to_multifilesink"
    elif command_type == "pipeline_app_yuv_jpeg":
        return f"{kwargs['format']}_{kwargs['width']}x{kwargs['height']}_{kwargs['framerate']}fps_yuv_jpeg_multifilesink"
    elif command_type == "camera_wayland_encode":
        return f"camera{kwargs['camera']}_wayland_{kwargs['width']}x{kwargs['height']}_{kwargs['framerate1']}fps_encode_{kwargs['framerate2']}fps_{kwargs['bitrate']}bps"
    elif command_type == "pipeline_app_encode_jpeg":
        return f"camera{kwargs['camera']}_encode_{kwargs['width']}x{kwargs['height']}_{kwargs['framerate']}fps_jpeg_multifilesink"
    elif command_type == "dual_camera_multifilesink":
        return f"camera{kwargs['camera1']}_{kwargs['width1']}x{kwargs['height1']}_{kwargs['framerate1']}fps_multifilesink_camera{kwargs['camera2']}_{kwargs['width2']}x{kwargs['height2']}_{kwargs['framerate2']}fps_multifilesink"
    elif command_type == "pipeline_app_wayland_jpeg":
        return f"camera{kwargs['camera']}_wayland_{kwargs['width']}x{kwargs['height']}_jpeg_multifilesink"
    elif command_type == "pipeline_app_video_image":
        ids = []
        for w, h in zip(kwargs["widths"], kwargs["heights"]):
            id_ = f"camera{kwargs['camera']}_video{w}x{h}_image{kwargs['img_width']}x{kwargs['img_height']}"
            ids.append(id_)
        return ids
    else:
        return None

def save_to_json(commands, command_ids):
    try:
        with open('commands.json', 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):  # In case file does not exist or is empty
        data = {}

    for cmd, cmd_id in zip(commands, command_ids):
        data[cmd_id] = cmd

    with open('commands.json', 'w') as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":

    # Example usage for file_conversion:
    input_path = "/data/3840_2160_H264_30fps.mp4"
    output_path = "/data/mux3_fs.mp4"
    gst_command = generate_gst_command("file_conversion", input_file=input_path, output_file=output_path)
    command_id = generate_command_id("file_conversion", input_file=input_path, output_file=output_path)
    save_to_json(gst_command, command_id)
    print(f"Command saved with ID: {command_id}")

    # Example usage for camera_output:
    gst_command = generate_gst_command("camera_output", camera=0, format="NV12", width=1920, height=1080, framerate="2/1", x=100, y=100, sink_width=960, sink_height=548)
    command_id = generate_command_id("camera_output", camera=0, format="NV12", width=1920, height=1080, framerate="2/1")
    save_to_json(gst_command, command_id)
    print(f"Command saved with ID: {command_id}")

    gst_command = generate_gst_command("pipeline_app", format="NV12", width=1920, height=1080, framerate="30/1", yuv_location="/data/output/framexd.yuv", jpg_location="/data/framexd.jpg")
    command_id = generate_command_id("pipeline_app", format="NV12", width=1920, height=1080, framerate="30/1")
    save_to_json(gst_command, command_id)
    print(f"Command saved with ID: {command_id}")

    gst_command = generate_gst_command("pipeline_app_yuv_jpeg", format="NV12", width=3848, height=2160, framerate="30/1", yuv_location="/data/output/frame%d.yuv", jpg_location="/data/frame%d.jpg")
    command_id = generate_command_id("pipeline_app_yuv_jpeg", format="NV12", width=3848, height=2160, framerate="30/1")
    save_to_json(gst_command, command_id)
    print(f"Command saved with ID: {command_id}")

    gst_command = generate_gst_command("camera_wayland_encode", camera=0, format="NV12", width=3840, height=2160, framerate1="30/1", framerate2="60/1", x=100, y=100, sink_width=960, sink_height=540, bitrate=10000000, output_file="/data/output/vid.mp4")
    command_id = generate_command_id("camera_wayland_encode", camera=0, format="NV12", width=3840, height=2160, framerate1="30/1", framerate2="60/1", bitrate=10000000)
    save_to_json(gst_command, command_id)
    print(f"Command saved with ID: {command_id}")

    gst_command = generate_gst_command("pipeline_app_encode_jpeg", camera=0, format="NV12", width=3840, height=2160, framerate="30/1", video_location="/data/vid.mp4", jpg_location="/data/camera@xd.jpg")
    command_id = generate_command_id("pipeline_app_encode_jpeg", camera=0, format="NV12", width=3840, height=2160, framerate="30/1")
    save_to_json(gst_command, command_id)
    print(f"Command saved with ID: {command_id}")

    gst_command = generate_gst_command("dual_camera_multifilesink", camera1=0, format1="NV12", width1=3840, height1=2160, framerate1="30/1", location1="/data/client1_framexd.yuv", camera2=1, format2="NV12", width2=3840, height2=2160, framerate2="38/1", location2="/data/client2_framexd.yuv")
    command_id = generate_command_id("dual_camera_multifilesink", camera1=0, width1=3840, height1=2160, framerate1="30/1", camera2=1, width2=3840, height2=2160, framerate2="38/1")
    save_to_json(gst_command, command_id)
    print(f"Command saved with ID: {command_id}")

    gst_command = generate_gst_command("pipeline_app_wayland_jpeg", camera=0, format="NV12", width=3840, height=2160, fullscreen="true", is_async="true", jpg_location="/data/framexd.jpg")
    command_id = generate_command_id("pipeline_app_wayland_jpeg", camera=0, width=3840, height=2160)
    save_to_json(gst_command, command_id)
    print(f"Command saved with ID: {command_id}")

    # gst_command = generate_gst_command("pipeline_app_video_image", camera=0, format="NV12", width=3840, height=2160, mp4_location="/data/mux_4k_avc.mp4", img_width=3848, img_height=2160, jpg_location="/data/frame%d.jpg")
    
    # command_id = generate_command_id("pipeline_app_video_image", camera=0, width=3840, height=2160, img_width=3848, img_height=2160)
    # save_to_json(gst_command, command_id)

    resolutions = [(3840, 2160), (1920, 1080)]
    gst_commands = generate_gst_command("pipeline_app_video_image", camera=0, format="NV12", widths=[res[0] for res in resolutions], heights=[res[1] for res in resolutions], mp4_location="/data/mux_4k_avc.mp4", img_width=3848, img_height=2160, jpg_location="/data/frame%d.jpg")
    command_ids = generate_command_id("pipeline_app_video_image", camera=0, widths=[res[0] for res in resolutions], heights=[res[1] for res in resolutions], img_width=3848, img_height=2160)
    save_to_json(gst_commands, command_ids)

