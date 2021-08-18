from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
from scipy.io import wavfile
from random import randint
from Bezier import Bezier
from osbpy import *

import pylab as plt
import numpy as np
import colorsys
import textwrap
import math
import os
import re

alt_file_name = {
    "#": "pound",
    "%": "percent",
    "&": "ampersand",
    "{": "leftCBracket",
    "}": "rightCBracket",
    "\\": "backSlash",
    "<": "leftABracket",
    ">": "rightABracket",
    "*": "asterisk",
    "?": "question",
    "/": "forward",
    " ": "blank",
    "$": "dollar",
    "!": "exclamation",
    ".": "period",
    "'": "single",
    '"': "double quotes",
    ":": "colon",
    "@": "at",
    "+": "plus",
    "`": "backtick",
    "|": "pipe",
    "=": "equal",
    " ": "space"
}
"""
.########.##.....##.##....##..######..########.####..#######..##....##..######.
.##.......##.....##.###...##.##....##....##.....##..##.....##.###...##.##....##
.##.......##.....##.####..##.##..........##.....##..##.....##.####..##.##......
.######...##.....##.##.##.##.##..........##.....##..##.....##.##.##.##..######.
.##.......##.....##.##..####.##..........##.....##..##.....##.##..####.......##
.##.......##.....##.##...###.##....##....##.....##..##.....##.##...###.##....##
.##........#######..##....##..######.....##....####..#######..##....##..######.
"""


# Functions
def create_character(char, font_size, font, font_color=(0, 0, 0)):
    char_image = Image.new('RGBA',
                           font[1].getsize(char),
                           color=(255, 255, 255, 0))
    char_draw = ImageDraw.Draw(char_image)
    char_draw.text((0, 0), char, font=font[1], fill=font_color)
    char_name = char if char not in alt_file_name else alt_file_name[char]
    char_name += 'u' if char.isupper() else 'l'
    char_image.save(
        f'SB\char\{font[0]}{font_size}{char_name}{font_color[0]}{font_color[1]}{font_color[2]}.png'
    )
    return f"SB\char\{font[0]}{font_size}{char_name}{font_color[0]}{font_color[1]}{font_color[2]}.png"


def create_text(text, font_size, font, font_color=(0, 0, 0), word_wrap=False):
    text_name = ''.join([
        alt_file_name[c] if c in alt_file_name else c.lower()
        for c in text[:10]
    ])
    text_name += f"{font_color[0]}{font_color[1]}{font_color[2]}"
    if word_wrap:
        lines = textwrap.wrap(text, width=word_wrap)
        char_image = Image.new(
            'RGBA', (max([font[1].getsize(line)[0] for line in lines]),
                     (len(lines) + 1) * font[1].getsize('0')[1]),
            color=(255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_image)

        for i, line in enumerate(lines):
            width, height = font[1].getsize(line)
            char_draw.text((0, (i + 1) * font[1].getsize('0')[1]),
                           line,
                           font=font[1],
                           fill=font_color)
        char_image.save(f'SB/text/{font[0]}{font_size}{text_name}.png')

    else:
        char_image = Image.new('RGBA',
                               font[1].getsize(text),
                               color=(255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_image)
        char_draw.text((0, 0), text, font=font[1], fill=font_color)
        char_image.save(f"SB/text/{font[0]}{font_size}{text_name}.png")
    return f"SB/text/{font[0]}{font_size}{text_name}.png"


def float_in_text(text,
                  start,
                  end,
                  postion,
                  time,
                  appear_time=40,
                  color=(0, 0, 0),
                  font_size=16,
                  fade_out=False):
    font = ('Arial', ImageFont.truetype("C:\Windows\Fonts\Arial.ttf",
                                        font_size))
    for (i, c), st, en in zip(
            enumerate(text),
            np.linspace(start,
                        start + (len(text) * appear_time),
                        len(text),
                        dtype=int),
        (np.linspace(start,
                     start + (len(text) * appear_time),
                     len(text),
                     dtype=int))[::-1]):
        st, en = int(st), int(en)
        if i % 2 == 0:
            char_obj = Osbject(create_character(c, font_size, font, color),
                               "Foreground", 'TopLeft',
                               postion[0] + (font[1].getsize(text[:i]))[0],
                               postion[1] + font_size)
            char_obj.movey(0, st, st + time, postion[1] + font_size,
                           postion[1])
            char_obj.fade(0, st, st + time, 0, 1)

            if fade_out:
                char_obj.fade(0, st + time, end - fade_out, 1, 1)
                char_obj.fade(0, end - fade_out, end, 1, 0)
            else:
                char_obj.fade(0, st + time, end - en - time, 1, 1)
                char_obj.movey(0, end - en - time, end - en, postion[1],
                               postion[1] - font_size)
                char_obj.fade(0, end - en - time, end - en, 1, 0)

        elif i % 2 != 0:
            char_obj = Osbject(create_character(c, font_size, font, color),
                               "Foreground", 'TopLeft',
                               postion[0] + (font[1].getsize(text[:i]))[0],
                               postion[1] - font_size)
            char_obj.movey(0, st, st + time, postion[1] - font_size,
                           postion[1])
            char_obj.fade(0, st, st + time, 0, 1)

            if fade_out:
                char_obj.fade(0, st + time, end - fade_out, 1, 1)
                char_obj.fade(0, end - fade_out, end, 1, 0)
            else:
                char_obj.fade(0, st + time, end - en - time, 1, 1)
                char_obj.movey(0, end - en - time, end - en, postion[1],
                               postion[1] + font_size)
                char_obj.fade(0, end - en - time, end - en, 1, 0)


def text_fade_in_text(text,
                      start,
                      end,
                      postion,
                      time,
                      color=(0, 0, 0),
                      font_size=16,
                      fade_out=False,
                      word_wrap=False):
    font = ('Arial', ImageFont.truetype("C:\Windows\Fonts\Arial.ttf",
                                        font_size))

    text_img = create_text(text,
                           font_size,
                           font,
                           font_color=color,
                           word_wrap=word_wrap)
    text_osb = Osbject(text_img, "Foreground", "Centre", postion[0],
                       postion[1])
    text_img = Image.open(text_img)

    text_osb.fade(0, start, start + time, 0, 1)

    if fade_out:
        text_osb.fade(0, start, end - fade_out, 1, 1)
        text_osb.fade(0, end - fade_out, end, 1, 0)
    else:
        text_osb.fade(0, start, end, 1, 1)


def circle_fade_in_text(text,
                        start,
                        end,
                        postion,
                        time,
                        color=(0, 0, 0),
                        font_size=16,
                        fade_out=False,
                        word_wrap=False):
    font = ('Arial', ImageFont.truetype("C:\Windows\Fonts\Arial.ttf",
                                        font_size))

    text_img = create_text(text,
                           font_size,
                           font,
                           font_color=color,
                           word_wrap=word_wrap)
    text_osb = Osbject(text_img, "Foreground", "Centre", postion[0],
                       postion[1])
    text_img = Image.open(text_img)

    circle_size = max(text_img.size)
    circle = Image.new("RGBA", (circle_size, circle_size),
                       color=(255, 255, 255, 0))

    circle_draw = ImageDraw.Draw(circle)
    circle_draw.ellipse((0, 0, circle_size, circle_size), fill=(255, 255, 255))

    circle_draw.ellipse((0, 0, circle_size, circle_size), fill=(255, 255, 255))
    circle.save("SB\circle_fade_in.png")
    circle_osb = Osbject("SB/circle_fade_in.png", "Foreground", "Centre",
                         postion[0], postion[1])

    if fade_out:
        text_osb.fade(0, start, end - fade_out, 1, 1)
        text_osb.fade(0, end - fade_out, end, 1, 0)
    else:
        text_osb.fade(0, start, end, 1, 1)
    circle_osb.scale(0, start, start + time, 1, 0)


def type_text(text,
              start,
              end,
              postion,
              time_char=200,
              caret_flick_time=400,
              color=(0, 0, 0),
              font_size=16,
              fade_out=False):
    font = ('Ti83Pluspc',
            ImageFont.truetype("C:\Windows\Fonts\Ti-83PL.ttf", font_size))
    text += " "

    caret = Image.new("RGBA", font[1].getsize('y'), color=color)
    caret.save(f'SB/caret{color[0]}{color[1]}{color[2]}.png')

    starting_times = np.linspace(start, start + (len(text) * time_char),
                                 len(text))
    for (i, c), st in zip(enumerate(text), starting_times):
        st = int(st)
        char_osb = Osbject(create_character(c, font_size, font,
                                            color), "Foreground", 'TopLeft',
                           postion[0] + (font[1].getsize(text[:i]))[0],
                           postion[1])

        if fade_out:
            char_osb.fade(0, st, end - fade_out, 1, 1)
            char_osb.fade(0, end - fade_out, end, 1, 0)
        else:
            char_osb.fade(0, st, end, 1, 1)

        if i < len(starting_times) - 1:
            caret_osb = Osbject(f'SB/caret{color[0]}{color[1]}{color[2]}.png',
                                "Foreground", "TopLeft",
                                postion[0] + (font[1].getsize(text[:i]))[0],
                                postion[1])
            caret_osb.fade(0, st, int(starting_times[i + 1]) - 1, 1, 1)
            caret_osb.fade(0,
                           int(starting_times[i + 1]) - 1,
                           int(starting_times[i + 1]), 1, 0)

    caret_osb = Osbject(f'SB/caret{color[0]}{color[1]}{color[2]}.png',
                        "Foreground", "TopLeft",
                        postion[0] + (font[1].getsize(text[:-1]))[0],
                        postion[1])
    cur_time = int(starting_times[-1])

    for t in range(1, 7):
        if t % 2 == 0:
            caret_osb.fade(0, cur_time, cur_time + 1, 1, 0)
            caret_osb.fade(0, cur_time + 1, cur_time + caret_flick_time, 0, 0)
        else:
            caret_osb.fade(0, cur_time, cur_time + 1, 0, 1)
            caret_osb.fade(0, cur_time + 1, cur_time + caret_flick_time, 1, 1)
        cur_time += caret_flick_time


def idle_movement(filename,
                  start,
                  end,
                  scale=1,
                  magnitude=1,
                  position=(320, 240),
                  movement_times=(100, 1000),
                  layer="Foreground",
                  origin="Centre"):
    idle_osb = Osbject(filename, layer, origin, position[0], position[1])

    idle_img = Image.open(filename)
    idle_osb.scale(0, start, end, scale, scale)
    cur_pos, cur_time, cur_rotate = (position[0], position[1]), start, 0
    while cur_time < end:
        n_mvm, n_time, n_rotate = (
            position[0] +
            randint(-int(((idle_img.size[0] / 2) * scale) * magnitude),
                    int(((idle_img.size[0] / 2) * scale) * magnitude)),
            position[1] +
            randint(-int(((idle_img.size[1] / 2) * scale) * magnitude),
                    int(((idle_img.size[1] / 2) * scale) * magnitude))
        ), int(n) if (n := cur_time +
                      randint(movement_times[0], movement_times[1])
                      ) < end else end, randint(-1, 1) * magnitude
        idle_osb.move(0, cur_time, n_time, cur_pos[0], cur_pos[1], n_mvm[0],
                      n_mvm[1])
        idle_osb.rotate(0, cur_time, n_time, cur_rotate, n_rotate)
        cur_pos, cur_time, cur_rotate = n_mvm, n_time, n_rotate
        if cur_time >= end:
            break
    return idle_osb


def audio_visulizer_data(audiofile,
                         start,
                         end,
                         effect_range=(0, 1),
                         listen_range=(0, 1),
                         magnitude=1):
    # converts audiofile
    frame_rate, snd = wavfile.read(audiofile)
    specgram, frequencies, t, im = plt.specgram(snd[:, 0],
                                                NFFT=1024,
                                                Fs=frame_rate,
                                                noverlap=5,
                                                mode='magnitude')
    # averages the peak amplitudes of the frequency between desired frequency
    specgram = np.average(
        specgram[int(len(specgram) *
                     listen_range[0]):int(len(specgram) * listen_range[1])],
        axis=0).tolist()
    # Highest and lowest points
    maximum = plt.amax(specgram)
    minimum = plt.amin(specgram)
    # Gives value an initial value, for the if statement to be able to check
    value = [
        math.ceil(
            (((specgram[0] - minimum) / (maximum - minimum)) *
             ((effect_range[1] - effect_range[0]) + effect_range[0])) * 1000) /
        1000
    ]
    time = []
    # Loops through the specgram ands adds the values that are in the designated time to the value and time list
    for index, power in enumerate(specgram):
        if (p := math.ceil(
            (((power - minimum) / (maximum - minimum)) *
             ((effect_range[1] - effect_range[0]) + effect_range[0])) * 1000) /
                1000) != value[-1] and int(
                    round(t[index] * 1000)) > start and int(
                        round(t[index] * 1000)) < end and index % 2 == 0:
            value.append(p)
            time.append(int(round(t[index] * 1000)))
    # Returns the value and time list, excluding the initial value of value as to align the two list correctly
    return value[1:], time


def fade_visualizer(audiofile,
                    start,
                    end,
                    magnitude=1,
                    effect_range=(0, 1),
                    brightener=None,
                    listen_range=(0, 1)):
    # Checks the data type of brightener as multiple types of data types can be sent through and proccessed/used
    if brightener == None:
        temp_img = Image.new('RGBA', (1920, 1080), color=(255, 255, 255))
        temp_img.save('SB/brigten.png')
        brightener = Osbject('SB/brigten.png', "Background", "Centre", 320,
                             256)
    elif type(brightener) == Osbject:
        pass
    elif type(brightener) == str:
        brightener = Osbject(brightener, "Background", "Centre", 320, 256)
    elif type(brightener) == tuple:
        temp_img = Image.new('RGBA', (1920, 1080), color=brightener)
        temp_img.save('SB/brigten.png')
        brightener = Osbject('SB/brigten.png', "Background", "Centre", 320,
                             256)
    else:
        raise ValueError(
            "Invalid brightener value, None, Osbject, Image Path file or RGB/RGBA Value Excepted"
        )

    value, time = audio_visulizer_data(audiofile,
                                       start,
                                       end,
                                       effect_range=effect_range,
                                       listen_range=listen_range)
    for p, (it, t) in zip(value, enumerate(time)):
        if it == 0:
            brightener.fade(0, start, t, p, p)
        elif it == len(time[:-1]):
            brightener.fade(0, time[it - 1], end, value[it - 1], p)
        else:
            brightener.fade(0, time[it - 1], t, value[it - 1], p)


def brigten_visualizer(audiofile,
                       start,
                       end,
                       effect_range=(0, 1),
                       brightener=None,
                       listen_range=(0, 1)):
    # Checks the data type of brightener as multiple types of data types can be sent through and proccessed/used
    if brightener == None:
        temp_img = Image.new('RGBA', (1920, 1080), color=(255, 255, 255))
        temp_img.save('SB/brigten.png')
        brightener = Osbject('SB/brigten.png', "Background", "Centre", 320,
                             256)
    elif type(brightener) == Osbject:
        pass
    elif type(brightener) == str:
        brightener = Osbject(brightener, "Background", "Centre", 320, 256)
    elif type(brightener) == tuple:
        temp_img = Image.new('RGBA', (1920, 1080), color=brightener)
        temp_img.save('SB/brigten.png')
        brightener = Osbject('SB/brigten.png', "Background", "Centre", 320,
                             256)
    else:
        raise ValueError(
            "Invalid brightener value, None, Osbject, Image Path file or RGB/RGBA Value Excepted"
        )

    value, time = audio_visulizer_data(audiofile,
                                       start,
                                       end,
                                       effect_range=effect_range,
                                       listen_range=listen_range)
    pre_rgb = (0, 0, 0)
    for p, (it, t) in zip(value, enumerate(time)):
        (r, g, b) = colorsys.hsv_to_rgb(0, 0, p)
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        pr, pg, pb = pre_rgb
        if it == 0:
            brightener.colour(0, start, t, r, g, b, r, g, b)
        elif it == len(time[:-1]):
            brightener.colour(0, time[it - 1], end, pr, pg, pb, r, g, b)
        else:
            brightener.colour(0, time[it - 1], t, pr, pg, pb, r, g, b)
        pre_rgb = (r, g, b)


def get_individual_diff():
    r_dict = defaultdict()
    for file in os.listdir():
        if file.endswith('.osu'):
            with open(file, 'r', encoding="utf-8") as osufile:
                print(file)
                read_file = osufile.read()
                r_dict[file] = {
                    'file': read_file,
                    'storyboard': '',
                    'movement_path': calculate_diff_movement(read_file)
                }
    return r_dict


def save_diff(diffs):
    for file in diffs:
        with open(file, 'w', encoding="utf-8") as f:
            splt = diffs[file]['file'].split('\n')
            osu_file = '\n'.join(
                splt[:splt.index('//Storyboard Layer 3 (Foreground)') +
                     1]) + '\n' + diffs[file]['storyboard'] + '\n' + '\n'.join(
                         splt[splt.index('//Storyboard Layer 4 (Overlay)'):])
            f.write(osu_file)
    return


def parametric_circle(t, xc, yc, R):
    x = xc + R * np.cos(t)
    y = yc + R * np.sin(t)
    return x, y


def inv_parametric_circle(x, xc, R):
    t = np.arccos((x - xc) / R)
    return t


def define_circle(p1, p2, p3):
    """
    Returns the center and radius of the circle passing the given 3 points.
    In case the 3 points form a line, returns (None, infinity).
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((cx, cy), radius)


def calculate_diff_movement(mapinfo, fr=50):
    fr = int(1000 / fr)
    map_split = mapinfo.split('\n')
    r_lst = []
    t_timing_lst = []
    i_timing_lst = []
    sv = float(
        ((map_split[map_split.index('[Difficulty]') + 5]).split(':'))[1])

    for l in map_split[map_split.index('[TimingPoints]') +
                       1:map_split.index('[Colours]')]:
        l_splt = l.split(',')
        if len(l_splt) < 8:
            pass
        elif int(l_splt[6]) == 1:
            t_timing_lst.append([int(l_splt[0]), float(l_splt[1])])
        elif int(l_splt[6]) == 0:
            i_timing_lst.append([int(l_splt[0]), -float(l_splt[1])])
    t_timing_lst, i_timing_lst = np.array(t_timing_lst), np.array(i_timing_lst)

    # (256,192):(320,256)|64
    for l in map_split[map_split.index('[HitObjects]') + 1:-1]:
        l_split = l.split(',')

        if int(l_split[3]) % 2 == 0 and l_split[3] != '12':
            slider_info = l_split[5].split('|')
            slider_st = int(l_split[2])
            last_inherited = i_timing_lst[i_timing_lst[:, 0] <= slider_st]
            last_timing = t_timing_lst[t_timing_lst[:, 0] <= slider_st]
            slider_len = float(l_split[7])
            repeats = int(l_split[6])
            ssv = (10000 /
                   (last_inherited[-1, 1] if len(last_inherited) != 0 else 100)
                   ) * sv
            msPerBeat = last_timing[-1, 1]
            slider_time = ((slider_len / ssv) * msPerBeat) * repeats
            s_times = np.linspace(0, slider_time, int(slider_len) * repeats)

            if slider_info[0] == 'L':
                x, y = int(l_split[0]), int(l_split[1])
                ex, ey = int(slider_info[1].split(':')[0]), int(
                    slider_info[1].split(':')[1])
                s_x = np.linspace(x, ex, int(slider_len))
                s_y = np.linspace(y, ey, int(slider_len))

                if repeats > 1:
                    for reap in range(repeats):
                        if reap % 2 != 0:
                            s_x = np.append(s_x, np.flip(s_x))
                            s_y = np.append(s_y, np.flip(s_y))
                        if reap % 2 == 0 and reap != 0:
                            s_x = np.append(s_x, s_x)
                            s_y = np.append(s_y, s_y)

                for index, t in np.ndenumerate(s_times):
                    r_lst.append([
                        slider_st + int(t),
                        (int(s_x[index]) + 64, int(s_y[index]) + 64)
                    ])

            elif slider_info[0] == 'P':
                slider_points = [[int(l_split[0]), int(l_split[1])]]
                slider_points += [[
                    int(sli.split(':')[0]),
                    int(sli.split(':')[1])
                ] for sli in slider_info[1:]]

                center, radius = define_circle(slider_points[0],
                                               slider_points[1],
                                               slider_points[2])

                start_t = np.rad2deg(
                    np.arctan2(slider_points[0][1] - center[1],
                               slider_points[0][0] - center[0]))

                end_t_1 = start_t + (360 * (slider_len / (2 * np.pi * radius)))
                end_t_2 = start_t - (360 * (slider_len / (2 * np.pi * radius)))
                center_point = np.rad2deg(
                    np.arctan2(slider_points[1][1] - center[1],
                               slider_points[1][0] - center[0]))
                arc_T_1 = np.linspace(start_t, end_t_1, int(slider_len))
                arc_T_2 = np.linspace(start_t, end_t_2, int(slider_len))

                closet_point_1 = (np.abs(arc_T_1 - center_point)).argmin()
                closet_point_2 = (np.abs(arc_T_2 - center_point)).argmin()
                arc_T = arc_T_1 if closet_point_1 != 0 and closet_point_1 != arc_T_1.size - 1 else arc_T_2
                if (closet_point_1 == 0 and closet_point_2 == arc_T_2.size -
                        1) or (closet_point_2 == 0
                               and closet_point_1 == arc_T_1.size - 1):
                    center_point = 360 + np.rad2deg(
                        np.arctan2(slider_points[1][1] - center[1],
                                   slider_points[1][0] - center[0]))
                    closet_point_1 = (np.abs(arc_T_1 - center_point)).argmin()
                    closet_point_2 = (np.abs(arc_T_2 - center_point)).argmin()
                    arc_T = arc_T_1 if closet_point_1 != 0 and closet_point_1 != arc_T_1.size - 1 else arc_T_2

                arc_points = [[
                    center[0] + (radius * np.cos(np.deg2rad(angle))),
                    center[1] + (radius * np.sin(np.deg2rad(angle)))
                ] for angle in arc_T]

                if repeats > 1:
                    for reap in range(repeats):
                        if reap % 2 != 0:
                            arc_points += arc_points[::-1]
                        if reap % 2 == 0 and reap != 0:
                            arc_points += arc_points

                for index, t in np.ndenumerate(s_times):
                    index = index[0]
                    r_lst.append([
                        slider_st + int(t),
                        (int(arc_points[index][0]) + 64,
                         int(arc_points[index][1]) + 64)
                    ])

            elif slider_info[0] == 'B':
                slider_points = [[int(l_split[0]), int(l_split[1])]]
                slider_points += [[
                    int(sli.split(':')[0]),
                    int(sli.split(':')[1])
                ] for sli in slider_info[1:]]
                slider_sections = []
                sli_sect = []

                for ind, point in enumerate(slider_points):
                    if ind == 0:
                        sli_sect = [point]
                    elif ind == len(slider_points[:-1]):
                        sli_sect += [point]
                        slider_sections.append(np.array(sli_sect))
                    elif point == slider_points[ind - 1]:
                        sli_sect += [point]
                        slider_sections.append(np.array(sli_sect))
                        sli_sect = [point]
                    else:
                        sli_sect += [point]

                slider_cords = []
                for points in slider_sections:
                    distance = math.ceil(
                        np.sqrt(((points[-1][0] - points[0][0])**2) +
                                ((points[-1][1] - points[0][1])**2)))
                    t_points = np.linspace(0, 1,
                                           distance if distance > 0 else 1)
                    slider_cords.append(Bezier.Curve(t_points, points))
                slider_cords = np.concatenate(tuple(slider_cords))
                # s_times = np.linspace(0, slider_time,
                #                       int(slider_len) * repeats)

                if repeats > 1:
                    for reap in range(repeats):
                        if reap % 2 != 0:
                            slider_cords = np.append(slider_cords,
                                                     np.flip(slider_cords,
                                                             axis=0),
                                                     axis=0)
                        if reap % 2 == 0 and reap != 0:
                            slider_cords = np.append(slider_cords,
                                                     slider_cords,
                                                     axis=0)

                for index, t in np.ndenumerate(s_times):
                    index = int(index[0] * (len(slider_cords) /
                                            (slider_len * repeats)))
                    r_lst.append([
                        slider_st + int(t),
                        (int(slider_cords[index][0]) + 64,
                         int(slider_cords[index][1]) + 64)
                    ])

            elif slider_info[0] == 'C':
                pass
        elif int(l_split[3]) % 2 != 0 and l_split[3] != '12':
            # circle
            r_lst.append([
                int(l_split[2]), (int(l_split[0]) + 64, int(l_split[1]) + 64)
            ])

        elif l_split[3] == '12':
            pass

        else:
            pass
    return np.array(r_lst, dtype=object)


"""
.####.##....##.########.########...#######.
..##..###...##....##....##.....##.##.....##
..##..####..##....##....##.....##.##.....##
..##..##.##.##....##....########..##.....##
..##..##..####....##....##...##...##.....##
..##..##...###....##....##....##..##.....##
.####.##....##....##....##.....##..#######.
"""
print('getting individual diffs')
individual_diffs = get_individual_diff()
# Intro (00:01:031 - 00:36:332)
'''

White Background
Text apears/float in saying
'Camellia - Witness of Eternity'
'this mapset was made by KoiSihu's Mapping Community and is dedicated to him. Thanks for all the fun content. Much Love <3
Ti83Pluspc
'Mappers:'
'
Hytaa
IGuiqik
Kreign
eleni-
yhoundz
Diversity 0
kyurem005
Randomina
naska_link
'

'Hitsounds:'
'Hytaa'

'Storyboard:'
'naska_link'

Then text
'Please Enjoy'

'''
# Intro Background
# white_bg = Image.new('RGB', (1920, 1080), color='white')
# white_bg.save('SB/whiteBG.png')
print('setting up idle bg')
intro_bg = Osbject("SB/apocalypse1_blurred.jpg", "Background", "Centre", 320,
                   240)
intro_bg.scale(0, 15, 36332, .45, .45)
intro_bg.fade(0, 15, 1031, 0, .8)
intro_bg.fade(0, 1031, 35070, .8, .8)
intro_bg.fade(0, 35070, 36332, .8, 0)
print('Intro Credits')
font_color = (255, 255, 255)
shadow_color = (0, 0, 0)
# Koi dedication
text_fade_in_text(
    "This mapset was made by KoiFishu's Mapping Community and is dedicated to him. Thank you for all the fun content. Much Love! <3",
    4796,
    35070, (322, 162),
    1500,
    font_size=18,
    fade_out=200,
    color=(0, 0, 0),
    word_wrap=40)

text_fade_in_text(
    "This mapset was made by KoiFishu's Mapping Community and is dedicated to him. Thank you for all the fun content. Much Love! <3",
    4796,
    35070, (320, 160),
    1500,
    font_size=18,
    fade_out=200,
    color=font_color,
    word_wrap=40)

# Song name
float_in_text('Camellia - Witness of Eternity',
              1031,
              35070, (86, 82),
              800,
              appear_time=80,
              fade_out=200,
              color=(0, 0, 0),
              font_size=36)

float_in_text('Camellia - Witness of Eternity',
              1031,
              35070, (84, 80),
              800,
              appear_time=80,
              fade_out=200,
              color=font_color,
              font_size=36)

# Credit Sections
mappers_pos = (65, 265)
type_text('Mappers:',
          7603,
          35070, (mappers_pos[0] + 2, mappers_pos[1] + 2),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=shadow_color,
          font_size=16)
type_text('Hitsounds:',
          13505,
          35070, (mappers_pos[0] + 2, mappers_pos[1] + 102),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=shadow_color,
          font_size=16)
type_text('Storyboard:',
          20313,
          35070, (mappers_pos[0] + 262, mappers_pos[1] + 102),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=shadow_color,
          font_size=16)

type_text('Mappers:',
          7603,
          35070,
          mappers_pos,
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=font_color,
          font_size=16)
type_text('Hitsounds:',
          13505,
          35070, (mappers_pos[0], mappers_pos[1] + 100),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=font_color,
          font_size=16)
type_text('Storyboard:',
          20313,
          35070, (mappers_pos[0] + 260, mappers_pos[1] + 100),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=font_color,
          font_size=16)
# Mappers Name
type_text('hytaa',
          10647,
          35070, (mappers_pos[0] + 10, mappers_pos[1] + 22),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=shadow_color,
          font_size=16)
type_text('Ayumi',
          10657,
          35070, (mappers_pos[0] + 10, mappers_pos[1] + 42),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=shadow_color,
          font_size=16)
type_text('Kreign',
          10667,
          35070, (mappers_pos[0] + 10, mappers_pos[1] + 62),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=shadow_color,
          font_size=16)
type_text('eleni-',
          10657,
          35070, (mappers_pos[0] + 182, mappers_pos[1] + 22),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=shadow_color,
          font_size=16)
type_text('yhoundz',
          10667,
          35070, (mappers_pos[0] + 182, mappers_pos[1] + 42),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=shadow_color,
          font_size=16)
type_text('Diversity 0',
          10677,
          35070, (mappers_pos[0] + 182, mappers_pos[1] + 62),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=shadow_color,
          font_size=16)
type_text('kyurem005',
          10667,
          35070, (mappers_pos[0] + 362, mappers_pos[1] + 22),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=shadow_color,
          font_size=16)
type_text('Randomina',
          10677,
          35070, (mappers_pos[0] + 362, mappers_pos[1] + 42),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=shadow_color,
          font_size=16)
type_text('naska_link',
          10687,
          35070, (mappers_pos[0] + 362, mappers_pos[1] + 62),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=shadow_color,
          font_size=16)

type_text('Hytaa',
          10647,
          35070, (mappers_pos[0] + 8, mappers_pos[1] + 20),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=font_color,
          font_size=16)
type_text('Ayumi',
          10657,
          35070, (mappers_pos[0] + 8, mappers_pos[1] + 40),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=font_color,
          font_size=16)
type_text('Kreign',
          10667,
          35070, (mappers_pos[0] + 8, mappers_pos[1] + 60),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=font_color,
          font_size=16)
type_text('eleni-',
          10657,
          35070, (mappers_pos[0] + 180, mappers_pos[1] + 20),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=font_color,
          font_size=16)
type_text('yhoundz',
          10667,
          35070, (mappers_pos[0] + 180, mappers_pos[1] + 40),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=font_color,
          font_size=16)
type_text('Diversity 0',
          10677,
          35070, (mappers_pos[0] + 180, mappers_pos[1] + 60),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=font_color,
          font_size=16)
type_text('kyurem005',
          10667,
          35070, (mappers_pos[0] + 360, mappers_pos[1] + 20),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=font_color,
          font_size=16)
type_text('Randomina',
          10677,
          35070, (mappers_pos[0] + 360, mappers_pos[1] + 40),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=font_color,
          font_size=16)
type_text('naska_link',
          10687,
          35070, (mappers_pos[0] + 360, mappers_pos[1] + 60),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=font_color,
          font_size=16)
# Storyboader and Hitsounder are
type_text('Hytaa',
          16481,
          35070, (mappers_pos[0] + 10, mappers_pos[1] + 122),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=shadow_color,
          font_size=16)
type_text('naska_link',
          23071,
          35070, (mappers_pos[0] + 270, mappers_pos[1] + 122),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=shadow_color,
          font_size=16)

type_text('Hytaa',
          16481,
          35070, (mappers_pos[0] + 8, mappers_pos[1] + 120),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=font_color,
          font_size=16)
type_text('naska_link',
          23071,
          35070, (mappers_pos[0] + 268, mappers_pos[1] + 120),
          time_char=35,
          caret_flick_time=400,
          fade_out=200,
          color=font_color,
          font_size=16)
"""
.########..##.....##.####.##.......########................##.....##.########.
.##.....##.##.....##..##..##.......##.....##...............##.....##.##.....##
.##.....##.##.....##..##..##.......##.....##...............##.....##.##.....##
.########..##.....##..##..##.......##.....##....#######....##.....##.########.
.##.....##.##.....##..##..##.......##.....##...............##.....##.##.......
.##.....##.##.....##..##..##.......##.....##...............##.....##.##.......
.########...#######..####.########.########.................#######..##.......
"""
# Build-Up (00:36:332 - 01:16:541)
'''

00:36:332
Transition to Background
Brighten on impacts
Idle background shake

00:56:436
Maybe Rain falling, getting faster
Until build up ends, then it clears

01:15:284
Zoom through the rain, to clear BG 

'''
print('BG Visual and Idle movement')
bg_osb = idle_movement('apocalypse1.jpeg',
                       36320,
                       315063,
                       scale=.47,
                       magnitude=.0065,
                       movement_times=(400, 1000),
                       layer='Background')
bg_osb.fade(0, 35070, 36320, 0, 1)
brigten_visualizer('audio.wav',
                   36320,
                   206907,
                   effect_range=(.4, 1),
                   brightener=bg_osb,
                   listen_range=(0, 1))
# spec_bars = Image.new('RGBA',(1,1),color=(255,255,255))
# spec_bars.save('SB/spect_bars.png')
# spectrum1 = spectrum('audio.wav',
#                      'SB/spect_bars.png',
#                      10,
#                      100,
#                      88,
#                      36320,
#                      146907,
#                      120,
#                      256,
#                      'Foreground',
#                      'BottomCentre',
#                      gap=1,
#                      arrange="")

# fade_visualizer('audio.wav',
#                                36320,
#                                146907,
#                                magnitude=.3,
#                                listen_range=(.1, .5))
"""
.##....##.####....###....####.......##......########.....###....########..########.......##..
.##...##...##....##.##....##......####......##.....##...##.##...##.....##....##........####..
.##..##....##...##...##...##........##......##.....##..##...##..##.....##....##..........##..
.#####.....##..##.....##..##........##......########..##.....##.########.....##..........##..
.##..##....##..#########..##........##......##........#########.##...##......##..........##..
.##...##...##..##.....##..##........##......##........##.....##.##....##.....##..........##..
.##....##.####.##.....##.####.....######....##........##.....##.##.....##....##........######
"""
# Kiai 1 part 1 (01:16:541 - 01:36:646)
print('Spectrum Thingy')
spect_bar = Image.new('RGB', (20, 1), color=(255, 255, 255))
spect_bar.save('SB/spect_bars.png')

for ind, spect in enumerate(l_spectrum := spectrum('audio.wav',
                                                   './SB/spect_bars.png',
                                                   30,
                                                   225,
                                                   21,
                                                   76227,
                                                   128059,
                                                   10,
                                                   240,
                                                   "Foreground",
                                                   "TopLeft",
                                                   gap=35)[::-1]):
    spect.rotate(0, 76227, 128059, -1.57, -1.57)
    y_place = 25 + (ind * 22)
    spect.move(0, 76227, 76541, -150, y_place, -100, y_place)
    spect.move(0, 76541, 126803, -100, y_place, -100, y_place)
    spect.move(0, 126803, 128059, -100, y_place, -150, y_place)

    (r, g, b) = colorsys.hsv_to_rgb((1 / len(l_spectrum)) * ind, .05, 1)
    r, g, b = int(255 * r), int(255 * g), int(255 * b)
    spect.colour(0, 76541, 126803, r, g, b, r, g, b)

    spect.fade(0, 76227, 76541, 0, 0.6)
    spect.fade(0, 76541, 126803, 0.6, 0.6)
    spect.fade(0, 126803, 128059, 0.6, 0)

for ind, spect in enumerate(r_spectrum := spectrum('audio.wav',
                                                   './SB/spect_bars.png',
                                                   30,
                                                   225,
                                                   21,
                                                   76227,
                                                   128059,
                                                   10,
                                                   240,
                                                   "Foreground",
                                                   "TopLeft",
                                                   gap=35)[::-1]):
    spect.rotate(0, 76227, 128059, 1.57, 1.57)
    y_place = 25 + (ind * 22)
    spect.move(0, 76227, 76541, 790, y_place, 740, y_place)
    spect.move(0, 76541, 126803, 740, y_place, 740, y_place)
    spect.move(0, 126803, 128059, 740, y_place, 790, y_place)

    (r, g, b) = colorsys.hsv_to_rgb((1 / len(r_spectrum)) * ind, .05, 1)
    r, g, b = int(255 * r), int(255 * g), int(255 * b)
    spect.colour(0, 76541, 126803, r, g, b, r, g, b)

    spect.fade(0, 76227, 76541, 0, 0.6)
    spect.fade(0, 76541, 126803, 0.6, 0.6)
    spect.fade(0, 126803, 128059, 0.6, 0)
# individual_diff
print('movement bars')
for diff in individual_diffs:
    mouse_mvmt = individual_diffs[diff]['movement_path']
    start_t = 76541
    end_t = 95389
    mask = (mouse_mvmt[:, 0] <= end_t) & (mouse_mvmt[:, 0] >= start_t)
    mask = mouse_mvmt[mask].tolist()

    sb_xbar = f'Sprite,Foreground,Centre,"SB\\bar.png",320,256\n_V,0,{start_t},{start_t},36,0.6,36,0.6\n_R,0,{start_t},{start_t},1.57,1.57\n_F,0,76227,{start_t},0,0.85\n_F,0,{start_t},{end_t},0.85,0.85\n_F,0,{end_t},96331,0.85,0\n'
    sb_ybar = f'Sprite,Foreground,Centre,"SB\\bar.png",320,256\n_V,0,{start_t},{start_t},36,0.6,36,0.6\n_F,0,76227,{start_t},0,0.85\n_F,0,{start_t},{end_t},0.85,0.85\n_F,0,{end_t},96331,0.85,0\n'
    for index, mvmtpoint in enumerate(mask):
        time = mvmtpoint[0]
        cords = mvmtpoint[1]
        if time < end_t and index != 0:
            p_cords = mask[index - 1][1]
            p_time = mask[index - 1][0]
            sb_xbar += f'_MX,0,{p_time},{time},{p_cords[0]},{cords[0]}\n'
            sb_ybar += f'_MY,0,{p_time},{time},{p_cords[1]},{cords[1]}\n'
        elif time >= end_t:
            sb_xbar += f'_MX,0,{p_time},{time},{p_cords[0]},{cords[0]}\n'
            sb_ybar += f'_MY,0,{p_time},{time},{p_cords[1]},{cords[1]}\n'
            sb_xbar += f'_MX,0,{time},96331,{cords[0]},320\n'
            sb_ybar += f'_MY,0,{time},96331,{cords[1]},256\n'
        elif index == 0:
            sb_xbar += f'_MX,0,76227,{time},320,{cords[0]}\n'
            sb_ybar += f'_MY,0,76227,{time},320,{cords[1]}\n'

    individual_diffs[diff]['storyboard'] += sb_xbar + sb_ybar

    start_t = 106070
    end_t = 126803
    mask = (mouse_mvmt[:, 0] <= end_t) & (mouse_mvmt[:, 0] >= start_t)
    mask = mouse_mvmt[mask].tolist()

    sb_xbar = f'Sprite,Foreground,Centre,"SB\\bar.png",320,256\n_V,0,{start_t},{start_t},36,0.6,36,0.6\n_R,0,{start_t},{start_t},1.57,1.57\n_F,0,105756,{start_t},0,0.85\n_F,0,{start_t},{end_t},0.85,0.85\n_F,0,{end_t},128059,0.85,0\n'
    sb_ybar = f'Sprite,Foreground,Centre,"SB\\bar.png",320,256\n_V,0,{start_t},{start_t},36,0.6,36,0.6\n_F,0,105756,{start_t},0,0.85\n_F,0,{start_t},{end_t},0.85,0.85\n_F,0,{end_t},128059,0.85,0\n'
    for index, mvmtpoint in enumerate(mask):
        time = mvmtpoint[0]
        cords = mvmtpoint[1]
        if time < end_t and index != 0:
            p_cords = mask[index - 1][1]
            p_time = mask[index - 1][0]
            sb_xbar += f'_MX,0,{p_time},{time},{p_cords[0]},{cords[0]}\n'
            sb_ybar += f'_MY,0,{p_time},{time},{p_cords[1]},{cords[1]}\n'
        elif time >= end_t:
            sb_xbar += f'_MX,0,{p_time},{time},{p_cords[0]},{cords[0]}\n'
            sb_ybar += f'_MY,0,{p_time},{time},{p_cords[1]},{cords[1]}\n'
            sb_xbar += f'_MX,0,{time},128059,{cords[0]},320\n'
            sb_ybar += f'_MY,0,{time},128059,{cords[1]},256\n'
        elif index == 0:
            sb_xbar += f'_MX,0,105756,{time},320,{cords[0]}\n'
            sb_ybar += f'_MY,0,105756,{time},320,{cords[1]}\n'
    '''
    01:36:646 (1) - 01:37:117 (2) - 01:37:588 (3) - 01:38:059 (4) - 01:38:530 (5) - 314
    01:39:159 (1) - 01:39:630 (2) - 01:40:101 (3) - 01:40:572 (4) - 01:41:043 (5) - 99159 99630 100101 100572 101043
    '''
    individual_diffs[diff]['storyboard'] += sb_xbar + sb_ybar

    for imp_i, imp_t in enumerate(
            imp_ts := [96646, 97117, 97588, 98059, 98530]):
        mask = (mouse_mvmt[:, 0] == imp_t)
        mask = mouse_mvmt[mask].tolist()
        imp_br = 140
        imp_c = mask[0][1]
        # center[0] + (radius * np.cos(np.deg2rad(angle))),center[1] + (radius * np.sin(np.deg2rad(angle)))
        if imp_t != imp_ts[-1]:
            for imp_ang in np.linspace(0, 360, randint(15, 25)):
                imp_r = imp_br * (100 / randint(95, 105))
                imp_ang = imp_ang + randint(-20, 20)
                imp_ex, imp_ey = imp_c[0] + (
                    imp_r * np.cos(np.deg2rad(imp_ang))), imp_c[1] + (
                        imp_r * np.sin(np.deg2rad(imp_ang)))
                imp_et = imp_t + 314 - randint(0, 25)
                imp_rt = np.deg2rad(imp_ang) * (10 / randint(8, 12))
                imp_sc = randint(10, 35)
                imp_mit = imp_ts[-2] + 314
                individual_diffs[diff][
                    'storyboard'] += f'Sprite,Foreground,Centre,"SB\pixel.png",{imp_c[0]},{imp_c[1]}\n_M,1,{imp_t},{imp_et},{imp_c[0]},{imp_c[1]},{imp_ex},{imp_ey}\n_M,1,{imp_et},{imp_mit},{imp_ex},{imp_ey},{imp_ex},{imp_ey}\n_M,0,{imp_mit},{imp_ts[-1]},{imp_ex},{imp_ey},{imp_c[0]},{imp_c[1]}\n_R,0,{imp_t},{imp_et},{imp_rt},{imp_rt}\n_F,0,{imp_t},{imp_et},0.5,0.5\n_S,0,{imp_t},{imp_et},{imp_sc},{imp_sc}\n'

    for imp_i, imp_t in enumerate(
            imp_tz := [99159, 99630, 100101, 100572, 101044]):
        mask = (mouse_mvmt[:, 0] == imp_t)
        mask = mouse_mvmt[mask].tolist()
        imp_br = 140
        # center[0] + (radius * np.cos(np.deg2rad(angle))),center[1] + (radius * np.sin(np.deg2rad(angle)))
        if imp_t != imp_tz[-1]:
            imp_c = mask[0][1]
            for imp_ang in np.linspace(0, 360, randint(15, 25)):
                imp_r = imp_br * (100 / randint(95, 105))
                imp_ang = imp_ang + randint(-20, 20)
                imp_ex, imp_ey = imp_c[0] + (
                    imp_r * np.cos(np.deg2rad(imp_ang))), imp_c[1] + (
                        imp_r * np.sin(np.deg2rad(imp_ang)))
                imp_et = imp_t + 314 - randint(0, 25)
                imp_rt = np.deg2rad(imp_ang) * (10 / randint(8, 12))
                imp_sc = randint(10, 35)
                imp_mit = imp_tz[-2] + 314
                individual_diffs[diff][
                    'storyboard'] += f'Sprite,Foreground,Centre,"SB\pixel.png",{imp_c[0]},{imp_c[1]}\n_M,1,{imp_t},{imp_et},{imp_c[0]},{imp_c[1]},{imp_ex},{imp_ey}\n_M,1,{imp_et},{imp_mit},{imp_ex},{imp_ey},{imp_ex},{imp_ey}\n_M,0,{imp_mit},{imp_tz[-1]},{imp_ex},{imp_ey},{imp_c[0]},{imp_c[1]}\n_R,0,{imp_t},{imp_et},{imp_rt},{imp_rt}\n_F,0,{imp_t},{imp_et},0.5,0.5\n_S,0,{imp_t},{imp_et},{imp_sc},{imp_sc}\n'
'''

01:16:541
X Y Line of cursor movement, to hit objects, and follow paths (This will be annoying)
Spectrums on the side
Brigten on bass hits

'''
"""
.##....##.####....###....####.......##......########.....###....########..########.....#######.
.##...##...##....##.##....##......####......##.....##...##.##...##.....##....##.......##.....##
.##..##....##...##...##...##........##......##.....##..##...##..##.....##....##..............##
.#####.....##..##.....##..##........##......########..##.....##.########.....##........#######.
.##..##....##..#########..##........##......##........#########.##...##......##.......##.......
.##...##...##..##.....##..##........##......##........##.....##.##....##.....##.......##.......
.##....##.####.##.....##.####.....######....##........##.....##.##.....##....##.......#########
"""
# Kiai 1 part 2 (01:36:646 - 01:46:698)
'''

01:36:646
Square hit highlightcoming from spinny square

'''
"""
.##....##.####....###....####.......##......########.....###....########..########.....#######.
.##...##...##....##.##....##......####......##.....##...##.##...##.....##....##.......##.....##
.##..##....##...##...##...##........##......##.....##..##...##..##.....##....##..............##
.#####.....##..##.....##..##........##......########..##.....##.########.....##........#######.
.##..##....##..#########..##........##......##........#########.##...##......##..............##
.##...##...##..##.....##..##........##......##........##.....##.##....##.....##.......##.....##
.##....##.####.##.....##.####.....######....##........##.....##.##.....##....##........#######.
"""
# Kiai 1 part 3 (01:46:698 - 02:06:803)
'''

01:46:698
Same as Kiai 1 part 1 (01:16:541 - 01:36:646)

'''
"""
.########..##.....##.####.##.......########..........########...#######..##......##.##....##
.##.....##.##.....##..##..##.......##.....##.........##.....##.##.....##.##..##..##.###...##
.##.....##.##.....##..##..##.......##.....##.........##.....##.##.....##.##..##..##.####..##
.########..##.....##..##..##.......##.....##.#######.##.....##.##.....##.##..##..##.##.##.##
.##.....##.##.....##..##..##.......##.....##.........##.....##.##.....##.##..##..##.##..####
.##.....##.##.....##..##..##.......##.....##.........##.....##.##.....##.##..##..##.##...###
.########...#######..####.########.########..........########...#######...###..###..##....##
"""
# Build-down (02:06:803 - 02:26:907)
'''

02:06:803
Brighten on hits
slowly diming

'''
"""
.########..########..########....###....##....##
.##.....##.##.....##.##.........##.##...##...##.
.##.....##.##.....##.##........##...##..##..##..
.########..########..######...##.....##.#####...
.##.....##.##...##...##.......#########.##..##..
.##.....##.##....##..##.......##.....##.##...##.
.########..##.....##.########.##.....##.##....##
"""
# Break (02:26:907 - 02:47:012)
# brigten_visualizer('audio.wav',
#                    206907,
#                    227012,
#                    effect_range=(.4, 1),
#                    brightener=bg_osb,
#                    listen_range=(0, 1))
bg_osb.fade(0, 146907, 148164, 1, .05)
bg_osb.fade(0, 165756, 167012, .05, 1)
bg_osb_mid = idle_movement('apocalypse1.jpeg',
                           148164,
                           168000,
                           scale=.47,
                           magnitude=.0065,
                           movement_times=(400, 1000),
                           layer='Background')
bg_osb_mid.fade(0, 148164, 149021, .1, .1)
for bg_flash_t in [149421, 154447, 159473, 164499]:
    bg_osb_mid.fade(2, bg_flash_t, bg_flash_t + 628, 0.9, .1)
'''

02:26:907
continue dim
Clouds, lightning on piano hits

'''
"""
..#######..##.....##.####.########.########....##.....##.########.##........#######..########..##....##
.##.....##.##.....##..##..##..........##.......###...###.##.......##.......##.....##.##.....##..##..##.
.##.....##.##.....##..##..##..........##.......####.####.##.......##.......##.....##.##.....##...####..
.##.....##.##.....##..##..######......##.......##.###.##.######...##.......##.....##.##.....##....##...
.##..##.##.##.....##..##..##..........##.......##.....##.##.......##.......##.....##.##.....##....##...
.##....##..##.....##..##..##..........##.......##.....##.##.......##.......##.....##.##.....##....##...
..#####.##..#######..####.########....##.......##.....##.########.########..#######..########.....##...
"""
# Quiet Melody (02:47:012 - 03:07:117)
brigten_visualizer('audio.wav',
                   227012,
                   315063,
                   effect_range=(.4, 1),
                   brightener=bg_osb,
                   listen_range=(0, 1))
'''

02:47:012
Starts to rain, and gain speed

'''
"""
.##.....##.########.##........#######..########..##....##
.###...###.##.......##.......##.....##.##.....##..##..##.
.####.####.##.......##.......##.....##.##.....##...####..
.##.###.##.######...##.......##.....##.##.....##....##...
.##.....##.##.......##.......##.....##.##.....##....##...
.##.....##.##.......##.......##.....##.##.....##....##...
.##.....##.########.########..#######..########.....##...
"""
# Melody (03:07:117 - 03:17:169)
'''

03:07:117
Speeds through rain, to clear background 
Hits brighten

'''
"""
.########..##.....##.####.##.......########..........##.....##.########......#######.
.##.....##.##.....##..##..##.......##.....##.........##.....##.##.....##....##.....##
.##.....##.##.....##..##..##.......##.....##.........##.....##.##.....##...........##
.########..##.....##..##..##.......##.....##.#######.##.....##.########......#######.
.##.....##.##.....##..##..##.......##.....##.........##.....##.##...........##.......
.##.....##.##.....##..##..##.......##.....##.........##.....##.##...........##.......
.########...#######..####.########.########...........#######..##...........#########
"""
# Build-Up (03:17:169 - 03:37:274)
'''

03:17:169
Start squares flowing in

'''
"""
.##....##.####....###....####.....#######.....########.....###....########..########.......##..
.##...##...##....##.##....##.....##.....##....##.....##...##.##...##.....##....##........####..
.##..##....##...##...##...##............##....##.....##..##...##..##.....##....##..........##..
.#####.....##..##.....##..##......#######.....########..##.....##.########.....##..........##..
.##..##....##..#########..##.....##...........##........#########.##...##......##..........##..
.##...##...##..##.....##..##.....##...........##........##.....##.##....##.....##..........##..
.##....##.####.##.....##.####....#########....##........##.....##.##.....##....##........######
"""
# Kiai 2 part 1 (03:37:274 - 03:57:379)
'''

03:37:274
X Y Cross points cursor movement
Square snake, snaking through
Spectrum on side 
brigtening on hits

03:52:352
Spaz X Y Cross point

03:54:866
Lock back on

'''
print('Spectrum Thingy Pasrt 2')
for ind, spect in enumerate(l2_spectrum := spectrum('audio.wav',
                                                    './SB/spect_bars.png',
                                                    30,
                                                    225,
                                                    21,
                                                    216960,
                                                    288897,
                                                    10,
                                                    240,
                                                    "Foreground",
                                                    "TopLeft",
                                                    gap=35)[::-1]):
    spect.rotate(0, 216960, 288897, -1.57, -1.57)
    y_place = 25 + (ind * 22)
    spect.move(0, 216960, 217274, -150, y_place, -100, y_place)
    spect.move(0, 217274, 287640, -100, y_place, -100, y_place)
    spect.move(0, 287640, 288897, -100, y_place, -150, y_place)

    (r, g, b) = colorsys.hsv_to_rgb((1 / len(l2_spectrum)) * ind, .05, 1)
    r, g, b = int(255 * r), int(255 * g), int(255 * b)
    spect.colour(0, 216960, 288897, r, g, b, r, g, b)

    spect.fade(0, 216960, 217274, 0, 0.6)
    spect.fade(0, 217274, 287640, 0.6, 0.6)
    spect.fade(0, 287640, 288897, 0.6, 0)

for ind, spect in enumerate(r2_spectrum := spectrum('audio.wav',
                                                    './SB/spect_bars.png',
                                                    30,
                                                    225,
                                                    21,
                                                    216960,
                                                    288897,
                                                    10,
                                                    240,
                                                    "Foreground",
                                                    "TopLeft",
                                                    gap=35)[::-1]):
    spect.rotate(0, 216960, 288897, 1.57, 1.57)
    y_place = 25 + (ind * 22)
    spect.move(0, 216960, 217274, 790, y_place, 740, y_place)
    spect.move(0, 217274, 287640, 740, y_place, 740, y_place)
    spect.move(0, 287640, 288897, 740, y_place, 790, y_place)

    (r, g, b) = colorsys.hsv_to_rgb((1 / len(r2_spectrum)) * ind, .05, 1)
    r, g, b = int(255 * r), int(255 * g), int(255 * b)
    spect.colour(0, 216960, 288897, r, g, b, r, g, b)

    spect.fade(0, 216960, 217274, 0, 0.6)
    spect.fade(0, 217274, 287640, 0.6, 0.6)
    spect.fade(0, 287640, 288897, 0.6, 0)
# individual_diff
print('movement bars part 2')
for diff in individual_diffs:
    mouse_mvmt = individual_diffs[diff]['movement_path']
    start_t = 217274
    end_t = 237379
    mask = (mouse_mvmt[:, 0] <= end_t) & (mouse_mvmt[:, 0] >= start_t)
    mask = mouse_mvmt[mask].tolist()

    sb_xbar = f'Sprite,Foreground,Centre,"SB\\bar.png",320,256\n_V,0,{start_t},{start_t},36,0.6,36,0.6\n_R,0,{start_t},{start_t},1.57,1.57\n_F,0,216960,{start_t},0,0.85\n_F,0,{start_t},{end_t},0.85,0.85\n_F,0,{end_t},238007,0.85,0\n'
    sb_ybar = f'Sprite,Foreground,Centre,"SB\\bar.png",320,256\n_V,0,{start_t},{start_t},36,0.6,36,0.6\n_F,0,216960,{start_t},0,0.85\n_F,0,{start_t},{end_t},0.85,0.85\n_F,0,{end_t},238007,0.85,0\n'
    for index, mvmtpoint in enumerate(mask):
        time = mvmtpoint[0]
        cords = mvmtpoint[1]
        if time < end_t and index != 0:
            p_cords = mask[index - 1][1]
            p_time = mask[index - 1][0]
            sb_xbar += f'_MX,0,{p_time},{time},{p_cords[0]},{cords[0]}\n'
            sb_ybar += f'_MY,0,{p_time},{time},{p_cords[1]},{cords[1]}\n'
        elif time >= end_t:
            sb_xbar += f'_MX,0,{p_time},{time},{p_cords[0]},{cords[0]}\n'
            sb_ybar += f'_MY,0,{p_time},{time},{p_cords[1]},{cords[1]}\n'
            sb_xbar += f'_MX,0,{time},238007,{cords[0]},320\n'
            sb_ybar += f'_MY,0,{time},238007,{cords[1]},256\n'
        elif index == 0:
            sb_xbar += f'_MX,0,216960,{time},320,{cords[0]}\n'
            sb_ybar += f'_MY,0,216960,{time},320,{cords[1]}\n'

    individual_diffs[diff]['storyboard'] += sb_xbar + sb_ybar

    start_t = 270049
    end_t = 287640
    mask = (mouse_mvmt[:, 0] <= end_t) & (mouse_mvmt[:, 0] >= start_t)
    mask = mouse_mvmt[mask].tolist()

    sb_xbar = f'Sprite,Foreground,Centre,"SB\\bar.png",320,256\n_V,0,{start_t},{start_t},36,0.6,36,0.6\n_R,0,{start_t},{start_t},1.57,1.57\n_F,0,269735,{start_t},0,0.85\n_F,0,{start_t},{end_t},0.85,0.85\n_F,0,{end_t},288897,0.85,0\n'
    sb_ybar = f'Sprite,Foreground,Centre,"SB\\bar.png",320,256\n_V,0,{start_t},{start_t},36,0.6,36,0.6\n_F,0,269735,{start_t},0,0.85\n_F,0,{start_t},{end_t},0.85,0.85\n_F,0,{end_t},288897,0.85,0\n'
    for index, mvmtpoint in enumerate(mask):
        time = mvmtpoint[0]
        cords = mvmtpoint[1]
        if time < end_t and index != 0:
            p_cords = mask[index - 1][1]
            p_time = mask[index - 1][0]
            sb_xbar += f'_MX,0,{p_time},{time},{p_cords[0]},{cords[0]}\n'
            sb_ybar += f'_MY,0,{p_time},{time},{p_cords[1]},{cords[1]}\n'
        elif time >= end_t:
            sb_xbar += f'_MX,0,{p_time},{time},{p_cords[0]},{cords[0]}\n'
            sb_ybar += f'_MY,0,{p_time},{time},{p_cords[1]},{cords[1]}\n'
            sb_xbar += f'_MX,0,{time},288897,{cords[0]},320\n'
            sb_ybar += f'_MY,0,{time},288897,{cords[1]},256\n'
        elif index == 0:
            sb_xbar += f'_MX,0,269735,{time},320,{cords[0]}\n'
            sb_ybar += f'_MY,0,269735,{time},320,{cords[1]}\n'
    '''
    01:36:646 (1) - 01:37:117 (2) - 01:37:588 (3) - 01:38:059 (4) - 01:38:530 (5) - 314
    01:39:159 (1) - 01:39:630 (2) - 01:40:101 (3) - 01:40:572 (4) - 01:41:043 (5) - 99159 99630 100101 100572 101043
    '''
    individual_diffs[diff]['storyboard'] += sb_xbar + sb_ybar

    for imp_i, imp_t in enumerate(
            imp_te := [257483, 257955, 258426, 258897, 259368]):
        mask = (mouse_mvmt[:, 0] == imp_t)
        mask = mouse_mvmt[mask].tolist()
        imp_br = 140
        # center[0] + (radius * np.cos(np.deg2rad(angle))),center[1] + (radius * np.sin(np.deg2rad(angle)))
        if imp_t != imp_te[-1]:
            imp_c = mask[0][1]
            for imp_ang in np.linspace(0, 360, randint(15, 25)):
                imp_r = imp_br * (100 / randint(95, 105))
                imp_ang = imp_ang + randint(-20, 20)
                imp_ex, imp_ey = imp_c[0] + (
                    imp_r * np.cos(np.deg2rad(imp_ang))), imp_c[1] + (
                        imp_r * np.sin(np.deg2rad(imp_ang)))
                imp_et = imp_t + 314 - randint(0, 25)
                imp_rt = np.deg2rad(imp_ang) * (10 / randint(8, 12))
                imp_sc = randint(10, 35)
                imp_mit = imp_te[-2] + 314
                individual_diffs[diff][
                    'storyboard'] += f'Sprite,Foreground,Centre,"SB\pixel.png",{imp_c[0]},{imp_c[1]}\n_M,1,{imp_t},{imp_et},{imp_c[0]},{imp_c[1]},{imp_ex},{imp_ey}\n_M,1,{imp_et},{imp_mit},{imp_ex},{imp_ey},{imp_ex},{imp_ey}\n_M,0,{imp_mit},{imp_te[-1]},{imp_ex},{imp_ey},{imp_c[0]},{imp_c[1]}\n_R,0,{imp_t},{imp_et},{imp_rt},{imp_rt}\n_F,0,{imp_t},{imp_et},0.5,0.5\n_S,0,{imp_t},{imp_et},{imp_sc},{imp_sc}\n'

    for imp_i, imp_t in enumerate(
            imp_tc := [259996, 260468, 260939, 261410, 261881]):
        mask = (mouse_mvmt[:, 0] == imp_t)
        mask = mouse_mvmt[mask].tolist()
        imp_br = 140
        # center[0] + (radius * np.cos(np.deg2rad(angle))),center[1] + (radius * np.sin(np.deg2rad(angle)))
        if imp_t != imp_tc[-1]:
            imp_c = mask[0][1]
            for imp_ang in np.linspace(0, 360, randint(15, 25)):
                imp_r = imp_br * (100 / randint(95, 105))
                imp_ang = imp_ang + randint(-20, 20)
                imp_ex, imp_ey = imp_c[0] + (
                    imp_r * np.cos(np.deg2rad(imp_ang))), imp_c[1] + (
                        imp_r * np.sin(np.deg2rad(imp_ang)))
                imp_et = imp_t + 314 - randint(0, 25)
                imp_rt = np.deg2rad(imp_ang) * (10 / randint(8, 12))
                imp_sc = randint(10, 35)
                imp_mit = imp_tc[-2] + 314
                individual_diffs[diff][
                    'storyboard'] += f'Sprite,Foreground,Centre,"SB\pixel.png",{imp_c[0]},{imp_c[1]}\n_M,1,{imp_t},{imp_et},{imp_c[0]},{imp_c[1]},{imp_ex},{imp_ey}\n_M,1,{imp_et},{imp_mit},{imp_ex},{imp_ey},{imp_ex},{imp_ey}\n_M,0,{imp_mit},{imp_tc[-1]},{imp_ex},{imp_ey},{imp_c[0]},{imp_c[1]}\n_R,0,{imp_t},{imp_et},{imp_rt},{imp_rt}\n_F,0,{imp_t},{imp_et},0.5,0.5\n_S,0,{imp_t},{imp_et},{imp_sc},{imp_sc}\n'
"""
.##....##.####....###....####.....#######.....##.....##.########.##........#######..########..##....##
.##...##...##....##.##....##.....##.....##....###...###.##.......##.......##.....##.##.....##..##..##.
.##..##....##...##...##...##............##....####.####.##.......##.......##.....##.##.....##...####..
.#####.....##..##.....##..##......#######.....##.###.##.######...##.......##.....##.##.....##....##...
.##..##....##..#########..##.....##...........##.....##.##.......##.......##.....##.##.....##....##...
.##...##...##..##.....##..##.....##...........##.....##.##.......##.......##.....##.##.....##....##...
.##....##.####.##.....##.####....#########....##.....##.########.########..#######..########.....##...
"""
# Kiai 2 Melody (03:57:379 - 04:17:483)
'''

03:57:379
Continue 

'''
"""
.##....##.####....###....####.....#######.....########.....###....########..########.....#######.
.##...##...##....##.##....##.....##.....##....##.....##...##.##...##.....##....##.......##.....##
.##..##....##...##...##...##............##....##.....##..##...##..##.....##....##..............##
.#####.....##..##.....##..##......#######.....########..##.....##.########.....##........#######.
.##..##....##..#########..##.....##...........##........#########.##...##......##.......##.......
.##...##...##..##.....##..##.....##...........##........##.....##.##....##.....##.......##.......
.##....##.####.##.....##.####....#########....##........##.....##.##.....##....##.......#########
"""
# Kiai 2 part 2 (04:17:483 - 04:30:049)
'''

04:17:483
Same as Kiai 1 part 2
Maybe with square snake, snaking through the hits

04:22:510
Square snake circles, slowling down towards end

04:27:693
The parts of the snake spin fast

04:29:735
Stops spinning, and changes to colors

'''
"""
.##....##.####....###....####.....#######.....########.....###....########..########.....#######.
.##...##...##....##.##....##.....##.....##....##.....##...##.##...##.....##....##.......##.....##
.##..##....##...##...##...##............##....##.....##..##...##..##.....##....##..............##
.#####.....##..##.....##..##......#######.....########..##.....##.########.....##........#######.
.##..##....##..#########..##.....##...........##........#########.##...##......##..............##
.##...##...##..##.....##..##.....##...........##........##.....##.##....##.....##.......##.....##
.##....##.####.##.....##.####....#########....##........##.....##.##.....##....##........#######.
"""
# Kiai 2 part 3 (04:30:049 - 04:47:640)
'''

04:30:049
The Squares are flying around going to hit objects, and follow slider path 
Also similar style to Kiai 2 part 1

04:46:384
All parts of snake freeze and shakes, like defeated

'''
"""
..#######..##.....##.########.########...#######.
.##.....##.##.....##....##....##.....##.##.....##
.##.....##.##.....##....##....##.....##.##.....##
.##.....##.##.....##....##....########..##.....##
.##.....##.##.....##....##....##...##...##.....##
.##.....##.##.....##....##....##....##..##.....##
..#######...#######.....##....##.....##..#######.
"""
# Outro (04:47:640 - 05:07:745)
'''

04:47:640
Snake parts slowly start to fall
and everything began to dim

05:07:745
One last square shows up spinning

'''
print('Crediting mapper')
individual_diffs[
    "Camellia - Witness of Eternity (Hytaa) [Diversity 0's Extreme].osu"][
        'storyboard'] += 'Sprite,Foreground,BottomLeft,"SB\\text\cooltext390285123963449.png",-100,452\n_S,0,0,315063,0.5,0.5\n'
individual_diffs["Camellia - Witness of Eternity (Hytaa) [eleni's Insane].osu"][
    'storyboard'] += 'Sprite,Foreground,BottomLeft,"SB\\text\cooltext390285071312598.png",-100,452\n_S,0,0,315063,0.5,0.5\n'
individual_diffs["Camellia - Witness of Eternity (Hytaa) [Naska's Normal].osu"][
    'storyboard'] += 'Sprite,Foreground,BottomLeft,"SB\\text\cooltext390285162647744.png",-100,452\n_S,0,0,315063,0.5,0.5\n'
individual_diffs["Camellia - Witness of Eternity (Hytaa) [Nathan's Hard].osu"][
    'storyboard'] += 'Sprite,Foreground,BottomLeft,"SB\\text\cooltext390285150071279.png",-100,452\n_S,0,0,315063,0.5,0.5\n'

#
'''Intro (00:01:031 - 00:36:332) - Diversity [DONE] (1031, 36332)
Build-Up (00:36:332 - 01:16:541) - hytaa [DONE] (36332, 76541)
Kiai 1 part 1 (01:16:541 - 01:36:646) - Kreign [DONE] (76541, 96646)
Kiai 1 part 2 (01:36:646 - 01:46:698) - yhoundz + Randomina [DONE] (96646, 106698)
Kiai 1 part 3 (01:46:698 - 02:06:803) - hytaa [DONE] (106698, 126803)
Build-down (02:06:803 - 02:26:907) - Ayumi [DONE] (126803, 146907)
Break (02:26:907 - 02:47:012) - yhoundz [DONE] (146907, 167012)
Quiet Melody (02:47:012 - 03:07:117) - eleni- [DONE] (167012, 187117)
Melody (03:07:117 - 03:17:169) - Randomina [DONE] (187117, 197169)
Build-Up (03:17:169 - 03:37:274) - Kreign [DONE] (197169, 217274)
Kiai 2 part 1 (03:37:274 - 03:57:379) - Diversity [DONE] (217274, 237379)
Kiai 2 Melody (03:57:379 - 04:17:483) - Ayumi [DONE] (237379, 257483)
Kiai 2 part 2 (04:17:483 - 04:30:049) - Randomina + yhoundz [DONE] (257483, 270049)
Kiai 2 part 3 (04:30:049 - 04:47:640) - eleni- [DONE] (270049, 287640)
Outro (04:47:640 - 05:07:745) - hytaa [DONE] (287640, 307745)
'''
for mapper_credit in [[(1031, 36332), 'cooltext390285123963449.png'],
                      [(36332, 76541), 'cooltext390284980029965.png'],
                      [(76541, 96646), 'cooltext390285058661646.png'],
                      [(96646, 106698), 'cooltext390286750564264.png'],
                      [(106698, 126803), 'cooltext390284980029965.png'],
                      [(126803, 146907), 'cooltext390285047015831.png'],
                      [(146907, 167012), 'cooltext390285107869736.png'],
                      [(167012, 187117), 'cooltext390285071312598.png'],
                      [(187117, 197169), 'cooltext390285150071279.png'],
                      [(197169, 217274), 'cooltext390285058661646.png'],
                      [(217274, 237379), 'cooltext390285123963449.png'],
                      [(237379, 257483), 'cooltext390285047015831.png'],
                      [(257483, 270049), 'cooltext390286764268423.png'],
                      [(270049, 287640), 'cooltext390285071312598.png'],
                      [(287640, 307745), 'cooltext390284980029965.png']]:
    individual_diffs[
        "Camellia - Witness of Eternity (Hytaa) [The Eternal Community].osu"][
            'storyboard'] += f'Sprite,Foreground,BottomLeft,"SB\\text\{mapper_credit[1]}",-100,452\n_F,0,{mapper_credit[0][0]},{mapper_credit[0][0]+628},0,1\n_F,0,{mapper_credit[0][1]-628},{mapper_credit[0][1]},1,0\n_S,0,{mapper_credit[0][0]},{mapper_credit[0][0]},0.5,0.5\n'

# Saves the storyboard
print('Saving')
Osbject.end("Camellia - Witness of Eternity (Hytaa).osb")
save_diff(individual_diffs)