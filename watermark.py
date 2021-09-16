from PIL import Image, ImageDraw, ImageFont
import sys
from io import BytesIO
import os
from pathlib import Path
from tqdm import tqdm
from DFLIMG.DFLJPG import DFLJPG
import numpy as np
import cv2
import multiprocessing as mp

image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
OUTPUT_FOLDER = 'watermarked images'
FONT_FAMILY = "arial.ttf"

def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)  # see below for Python 2.x
        else:
            yield entry

def get_image_paths(dir_path, image_extensions=image_extensions, subdirs=False, return_Path_class=False):
    dir_path = Path (dir_path)

    result = []
    if dir_path.exists():

        if subdirs:
            gen = scantree(str(dir_path))
        else:
            gen = os.scandir(str(dir_path))

        for x in list(gen):
            if any([x.name.lower().endswith(ext) for ext in image_extensions]):
                result.append( x.path if not return_Path_class else Path(x.path) )
    return sorted(result)

def value_of_pixel(pil_image, x, y):
    return pil_image.getpixel((x, y))

def get_text_dimensions(text_string, font):
    _, descent = font.getmetrics()

    text_width = font.getmask(text_string).getbbox()[2]
    text_height = font.getmask(text_string).getbbox()[3] + descent

    return (text_width, text_height)

def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                 .swapaxes(1, 2)
                 .reshape(-1, nrows, ncols))

def process_image(filepath):
    # check if image is an dflimg
    input_dfl = DFLJPG.load(filepath)
    is_dfl = True

    if not input_dfl or not input_dfl.has_data():
        print('\t################ No landmarks in file')
        is_dfl = False
        
    if is_dfl:
        dfl_data = input_dfl.get_dict()
        landmarks = input_dfl.get_landmarks()
        if input_dfl.has_seg_ie_polys() : xseg_polys = input_dfl.get_seg_ie_polys()
        xseg = input_dfl.get_xseg_mask_compressed()
        image = cv2.imdecode(xseg, cv2.IMREAD_UNCHANGED)
        image = image.astype(np.uint8)

        image_pil = Image.fromarray(image)
        w, h, _ = input_dfl.get_shape()

        # scale font size considering that 16 is good for 198res images
        font_size = int((16 * w) / 198)

        def_font = ImageFont.truetype(FONT_FAMILY, font_size)
        text_w, text_h = get_text_dimensions(f"{filepath.stem}", def_font)

        image_pil = image_pil.resize((w, h))

        matrix = []
        row = []

        for x in range(w):
            for y in range(h):
                if value_of_pixel(image_pil, x, y) == 255:
                    row.append(1)
                else:
                    row.append(0)
            matrix.append(row.copy())
            row.clear()

        np_matrix = np.array(matrix)

        # divide matrix in 4 matrix
        upper_left, upper_right, bottom_left, bottom_right = split(np_matrix, int(w/2), int(w/2))

        # possible positions strings
        possible_positions = ['upper_left', 'upper_right', 'bottom_left', 'bottom_right']

        # check sum of single matrix
        quadrants_value = {
            possible_positions[0] : np.sum(upper_left),
            possible_positions[1] : np.sum(upper_right),
            possible_positions[2] : np.sum(bottom_left),
            possible_positions[3] : np.sum(bottom_right)
        }

        possible_text_position = {
            possible_positions[0] : (2, 2),
            possible_positions[1] : ((w - text_w) - 2, 2),
            possible_positions[2] : (2, (h - text_h) - 2),
            possible_positions[3] : ((w - text_w) - 2, (h - text_h) - 2)
        }

        # choose best text position
        cord = possible_text_position[min(quadrants_value, key=quadrants_value.get)]

        final_image = Image.open(filepath)
        editable_image = ImageDraw.Draw(final_image)
        editable_image.text(cord, f"{filepath.stem}", (255, 0, 221), font=def_font)
        img_byte_arr = BytesIO()
        final_image.save(img_byte_arr, format='jpeg', quality=100, subsampling=0)

        # open again saved file and put dfl data inside
        OutputDflImg = DFLJPG.load(f'.\\{OUTPUT_FOLDER}\\{filepath.name}', image_as_bytes=img_byte_arr.getvalue())
        OutputDflImg.set_dict(dfl_data)
        OutputDflImg.set_landmarks(landmarks)
        if input_dfl.has_seg_ie_polys() : OutputDflImg.set_seg_ie_polys(xseg_polys)
        OutputDflImg.save()

    else:
        image = Image.open(filepath)
        editable_image = ImageDraw.Draw(image)
        editable_image.text((5,5), f"{filepath.stem}", (255, 0, 221))
        image.save(f'.\\{OUTPUT_FOLDER}\\{filepath.name}', quality=100, subsampling=0)

def input_int(s, default_value, valid_range=None, valid_list=None, add_info=None, show_default_value=True, help_message=None):
        if show_default_value:
            if len(s) != 0:
                s = f"[{default_value}] {s}"
            else:
                s = f"[{default_value}]"

        if add_info is not None or \
           valid_range is not None or \
           help_message is not None:
            s += " ("

        if valid_range is not None:
            s += f" {valid_range[0]}-{valid_range[1]}"

        if add_info is not None:
            s += f" {add_info}"

        if help_message is not None:
            s += " ?:help"

        if add_info is not None or \
           valid_range is not None or \
           help_message is not None:
            s += " )"

        s += " : "

        while True:
            try:
                inp = input(s)
                if len(inp) == 0:
                    raise ValueError("")

                if help_message is not None and inp == '?':
                    print (help_message)
                    continue

                i = int(inp)
                if valid_range is not None:
                    i = int(np.clip(i, valid_range[0], valid_range[1]))

                if (valid_list is not None) and (i not in valid_list):
                    i = default_value

                result = i
                break
            except:
                result = default_value
                break
        print (result)
        return result

# Module multiprocessing is organized differently in Python 3.4+
try:
    # Python 3.4+
    if sys.platform.startswith('win'):
        import multiprocessing.popen_spawn_win32 as forking
    else:
        import multiprocessing.popen_fork as forking
except ImportError:
    pass

if sys.platform.startswith('win'):
    # First define a modified version of Popen.
    class _Popen(forking.Popen):
        def __init__(self, *args, **kw):
            if hasattr(sys, 'frozen'):
                # We have to set original _MEIPASS2 value from sys._MEIPASS
                # to get --onefile mode working.
                os.putenv('_MEIPASS2', sys._MEIPASS)
            try:
                super(_Popen, self).__init__(*args, **kw)
            finally:
                if hasattr(sys, 'frozen'):
                    # On some platforms (e.g. AIX) 'os.unsetenv()' is not
                    # available. In those cases we cannot delete the variable
                    # but only set it to the empty string. The bootloader
                    # can handle this case.
                    if hasattr(os, 'unsetenv'):
                        os.unsetenv('_MEIPASS2')
                    else:
                        os.putenv('_MEIPASS2', '')

    # Second override 'Popen' class with our modified version.
    forking.Popen = _Popen

def main():
    if len(sys.argv) < 2:
        print('Wrong script usage. Correct usage:\n\tpython watermark.py <path of image folder')
        input('Press one key to continue . . .')
        exit(1)

    input_path = Path(sys.argv[1])

    debug = False
    if len(sys.argv) == 3:
        if sys.argv[2].lower() == 'debug':
            debug = True

    if not input_path.exists():
        print("input_dir not found.")
        return

    cpus = input_int('Insert number of CPUs to use',
                    help_message='If the default option is selected it will use all cpu cores and it will slow down pc',
                    default_value=mp.cpu_count())
    
    image_paths = [ Path(filepath) for filepath in get_image_paths(input_path) ]

    # create output folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    if debug:
        process_image(image_paths[0])
    else:
        with mp.Pool(processes=cpus) as p:
            list(tqdm(p.imap_unordered(process_image, image_paths), total=len(image_paths), ascii=True))

if __name__ == "__main__":
    # make the program starts with --onefile conf. pyinstaller on Windows system
    mp.freeze_support()
    main()
    input('Press a key to continue . . .')
