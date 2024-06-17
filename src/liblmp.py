import pandas as pd
import cv2
import numpy as np

led_kinds = ["_B", "_G", "_R", "_A"]

def parse_lmp(layout_list: list[list[str]]) -> dict[str, pd.DataFrame]:
    idx = 0
    layout_data = {}
    while idx < len(layout_list):
        col = layout_list[idx]
        if col != [] and col[0] == "T":
            table_name: str = col[1]
            table = []
            idx += 1
            while idx < len(layout_list) and layout_list[idx] != []:
                table.append(layout_list[idx])
                idx += 1
            layout_data[table_name] = pd.DataFrame(table[1:], columns=table[0])
            continue
        idx += 1
    return layout_data

def export_lmp6r(data: dict[str, pd.DataFrame]) -> str:
    tables = ["OPTION", "PATTERN", "MOVIE", "SOUND", "LAYER", "CLIP", "GUIDE", "SEQUENCE", "TRACK", "PTN_CLIP", "SEQ_MOVIE", "SEQ_SOUND", "SEQ_GUIDE", "SEQ_LOOPSTART", "NEXTUNIQUEID"]
    ret = "LMPP6\t4\n#\tlmpwork Pattern File\n\n"
    for t in tables:
        ret += "T\t" + t + "\n"
        ret += data[t].to_csv(sep="\t", index=False)
        ret += "\n"
    return ret

def sample_led(pixel: any, postfix: str) -> int:
    value = 0
    if postfix == "_B":
        value = pixel[0]
    if postfix == "_G":
        value = pixel[1]
    if postfix == "_R":
        value = pixel[2]
    if postfix == "_A":
        value = 255
    return int(value)

def mean_sample_led(frame: cv2.typing.MatLike, x: int, y: int, postfix: str) -> int:
    d = 4
    pixel = np.zeros((3))
    coeff_sum = 0
    for dx in range(-d, d + 1):
        for dy in range(-d, d + 1):
            distance = np.sqrt((dx ** 2) + (dy ** 2))
            coeff = 1 / np.sqrt(2 * np.pi) * np.exp(- (distance ** 2) / 2)
            pixel += frame[y + dy, x + dx] * coeff
            coeff_sum += coeff
    pixel /= coeff_sum
    return sample_led(pixel, postfix)
