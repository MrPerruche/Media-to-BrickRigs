import brci  # Brick Rigs Creation Interface
import cv2  # OpenCV2 - Video Processing
import librosa  # Librosa - .mp3 Audio Processing
from moviepy.editor import VideoFileClip  # MoviePy - Get audio from video
import os  # Os - Paths
from PIL import Image  # Pillow - Image Processing
import numpy as np  # Numpy weeeee


cwd = os.path.realpath(os.path.dirname(__file__))
in_project_dir = os.path.join(cwd, 'projects')

project_name = input('Enter your project\n> ')

in_cur_project_dir = os.path.join(cwd, 'projects', project_name)
images_dir = os.path.join(in_cur_project_dir, 'images')

data = brci.BRCI()

data.project_name = project_name
data.project_folder_directory = os.path.join(cwd, 'projects')

data.project_display_name = project_name.replace("-", " ").replace("_", " ").title()


def get_mode():

    while True:

        input_mode = input('[1] Generate Video\n[2] Record Audio Timestamps\n> ')

        if input_mode in ['1', '2']:

            return input_mode

        print('WARNING: Invalid Input')


def extract_audio_and_process(vid_path):
    video_name = os.path.splitext(os.path.basename(vid_path))[0]
    audio_file = f"audio_{video_name}.wav"

    # Extract audio from the video file
    video_clip = VideoFileClip(vid_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_file)

    # Load the extracted audio file
    y, sr = librosa.load(audio_file)

    # Analyze the audio to detect segments under a threshold
    threshold = 2  # TODO
    segments = librosa.effects.split(y, top_db=threshold)

    # Calculate pitch and length for each segment
    result = []
    for segment in segments:
        timestamp = segment[0] / sr  # Convert sample index to time
        length = (segment[1] - segment[0]) / sr  # Calculate length in seconds

        # Pass the audio data and segment boundaries to the pitch_calculation function
        pitch = pitch_calculation(segment, y, sr)

        result.append((timestamp, length, pitch))

    return result


# Ensure the pitch_calculation function is defined before calling it
def pitch_calculation(segment, audio_data, sr_):
    """
    Calculates the pitch for a given audio segment.

    Parameters:
    - segment: Tuple containing the start and end indices of the segment.
    - audio_data: The audio data array loaded using librosa.load().

    Returns:
    - pitch: An estimated pitch value for the segment.
    """
    # Extract the audio data for the current segment
    segment_audio = audio_data[segment[0]:segment[1]]

    # Estimate the fundamental frequency (F0) for the segment
    f0, _ = librosa.piptrack(y=segment_audio, sr=sr_)

    # Find the maximum F0 value among all detected pitches
    if len(f0) > 0:
        max_f0 = np.max(f0)
        # Normalize max_f0 to a scale where 1 represents 440 Hz
        normalized_max_f0 = max_f0 / 440
        # Scale the normalized value to fit within the range [0.5, 2]
        pitch = 0.5 + (normalized_max_f0 * 1.5)
    else:
        pitch = None

    return pitch


video_audio = []


def extract_frames(images_dir_, frame_rate, audio_, definition):

    global video_audio

    video_extensions = ['.mp4', '.avi', '.mov']  # Add more video extensions as needed
    image_extensions = ['.jpg', '.jpeg', '.png']  # Add more image extensions as needed

    video_file = next((f for f in os.listdir(images_dir_) if any(f.endswith(ext) for ext in video_extensions)), None)

    if not video_file:
        print("No video file found in the directory.")
        return

    if audio_:
        video_audio = extract_audio_and_process(os.path.join(images_dir_, video_file))

    print(video_audio)

    video_path = os.path.join(images_dir_, video_file)

    # Delete existing images in the directory
    for f in os.listdir(images_dir_):
        if any(f.endswith(ext) for ext in image_extensions):
            os.remove(os.path.join(images_dir_, f))

    os.makedirs(images_dir_, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_framerate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % int(video_framerate / frame_rate) == 0:
            resized_frame = cv2.resize(frame, definition)
            # Format the frame number with leading zeros
            frame_number_str = str(frame_count).zfill(4)
            cv2.imwrite(os.path.join(images_dir_, f"frame_{frame_number_str}.jpg"), resized_frame)

    cap.release()
    cv2.destroyAllWindows()


def compress_grid(grid):
    compressed_list = []
    rows = len(grid)
    cols = len(grid[0])

    for i in range(rows):
        for j in range(cols):
            if grid[i][j]:
                size_x = 1
                size_y = 1
                while j + size_x < cols and grid[i][j + size_x]:
                    size_x += 1
                while i + size_y < rows:
                    valid = True
                    for k in range(size_x):
                        if not grid[i + size_y][j + k]:
                            valid = False
                            break
                    if valid:
                        size_y += 1
                    else:
                        break
                compressed_list.append((j, i, size_x, size_y))

                for x in range(i, i + size_y):
                    for y in range(j, j + size_x):
                        grid[x][y] = False
    return compressed_list


def convert_images(bl_thresold: float = 0.18, invert_bl: bool = False, transposed: bool = True) -> list[list[list[bool]]]:

    def process_image(image_path_):
        image = Image.open(image_path_)
        bw_image = image.convert('L')
        pixels = list(bw_image.getdata())

        thresholded_pixels = [[invert_bl if pixel <= (256 * bl_thresold)-1 else (not invert_bl) for pixel in row] for row in chunks(pixels, bw_image.width)]
        return thresholded_pixels

    # Function to create chunks of data
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    # List to store nested lists for each image
    nested_lists_images = []

    # Iterate over each image in the directory
    for filename in sorted(os.listdir(images_dir)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(images_dir, filename)
            nested_lists_image = process_image(image_path)
            nested_lists_images.append(nested_lists_image)

    if transposed:
        nested_lists_images = list(map(list, zip(*nested_lists_images)))
        nested_lists_images = list(reversed(nested_lists_images))

    return nested_lists_images


def build_video(image_list: list[list[list[bool]]], audio: list[tuple], framerate: float = 4.0, scale: int = 1):

    if scale != 1:

        raise NotImplementedError('No scalable lights yet! Wait for 1.7.0...')

    height = len(image_list)
    frames = len(image_list[0])
    width = len(image_list[0][0])

    rom: list[list[tuple]] = []

    for image in image_list:

        rom += [compress_grid(image)]


    # Put a brick to build on
    main_support_length = frames + 3
    main_support_width = width + 2


    data.anb('main_support', 'ScalableBrick', {
        'BrickSize': [main_support_length, main_support_width, 1]
    }, [0+(main_support_length/0.2), 0+(main_support_width/0.2), 5])

    # Put a switch to turn it on
    data.anb('toggle', 'Switch_1sx1sx1s', {
        'bReturnToZero': False
    }, [0, 5, 15], [0, -90, 0])

    # Change Frames
    data.anb('piston', 'Actuator_1sx1sx2s_Bottom', {
        'MinLimit': 0,
        'MaxLimit': (frames*10 + 10) - 20,
        'SpeedFactor': framerate / 4.8,
        'BrickMaterial': 'Glass',
        'InputChannel': brci.BrickInput('Custom', ['toggle'])
    }, [-9, 5, 25], [0, -90, 0])

    # Change Frames B
    data.anb('piston_head', 'Actuator_1sx1sx2s_Top', {},
             [10, 5, 25], [0, -90, 0])

    # Support for sensors
    reader_support_height = (height-1)*3

    data.anb('reader_support', 'ScalableBrick', {
        'BrickSize': [1, 1, reader_support_height]
    }, [15, 5, 30+(reader_support_height/0.2)])

    # Support for ROM
    rom_support_height = (height-1)*3 + 4

    data.anb('rom_support', 'ScalableBrick', {
        'BrickSize': [1, 1, rom_support_height]
    }, [25, width*10 + 15, 10+(rom_support_height/0.2)])

    # Building video
    for row in range(height):

        # ROM Supports
        data.anb(f'rom_support_{row}', 'ScalableBrick', {
            'BrickSize': [frames+1, width+1, 1],
            'BrickColor': [0, 0, 32, 255],
            'ConnectorSpacing': [0, 3, 3, 3, 3, 0]
        }, [20+((frames+1)/0.2), 0+((width+1)/0.2), 45+(row*30)])

        # ROM Reader & Display
        for sensor in range(width):

            sens_pos = [25,
                        5 + 10*sensor,
                        20 + 30*row]

            light_pos = [-5,
                         5 + 10*sensor,
                         10*row]

            # Sensor
            data.anb(f'sensor_{row}_{sensor}', 'Sensor_1sx1sx1s', {
                'OutputChannel.MinIn': 0.02,
                'OutputChannel.MaxIn': 0.08,
                'OutputChannel.MinOut': 1.0,
                'OutputChannel.MaxOut': 0.0,
                'SensorType': 'Proximity',
                'TraceMask': 'Vehicles',
                'bReturnToZero': False
            }, sens_pos)

            # Its display
            data.anb(f'light_{row}_{sensor}', 'Light_1sx1sx1s', {
                'Brightness': 0.006,
                'LightDirection': 'Omnidirectional',
                'BrickMaterial': 'CloudyGlass',
                'InputChannel': brci.BrickInput('Custom', [f'sensor_{row}_{sensor}'])
            }, light_pos)

    # For each row (image)
    for i, row in enumerate(rom):

        # For each brick to be built in this row
        for brick in row:

            rom_pos = [30 + ((brick[1])*10) + (brick[3]/0.2),
                       0 + ((brick[0])*10) + (brick[2]/0.2),
                       35 + (i*30)]

            data.anb(f'rom_data_{row}_{brick}', 'ScalableBrick', {
                'BrickSize': [brick[3], brick[2], 1]
            }, rom_pos)





mode = get_mode()

match mode:

    # GENERATE VIDEO
    case '1':

        if not os.path.exists(os.path.join(in_cur_project_dir, 'images')):

            raise FileNotFoundError('The image folder is missing!')

        # Print or use nested_lists_images as needed
        source = input('[1] Convert a video to images and use these images\n[2] Use already inserted images\n[3] Convert a single image\n> ')


        valid_input: bool = False

        while not valid_input:
            match source:
                case '1':

                    image_framerate = float(input('How many images should be taken per second?\n> '))
                    res_x = int(input('Enter desired resolution X\n> '))
                    res_y = int(input('Enter desired resolution Y\n> '))

                    video_audio = []

                    extract_frames(images_dir, image_framerate, True, (res_x, res_y))

                    valid_input = True

                case '2':

                    valid_input = True

                case _:
                    print('WARNING: Invalid Input')

        images = convert_images(float(input('Enter desired B/L Threshold\n> ')), bool(input('To invert black and white, type anything before pressing enter.\n> ')))

        print(f"{video_audio=}")

        build_video(images, video_audio, float(input('Enter desired in-game framerate\n> ')))

        data.write_preview()
        data.write_metadata()
        data.write_brv()
        data.write_to_br()

    # RECORD TIMESTAMPS
    case '2':

        print('Mode 2')


    case '3':

        print('Mode 3')