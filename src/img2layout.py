import csv
import pandas as pd
import cv2

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

layout: str = "../layout/Unicorn.lmpl6r"
empty_pattern: str = "../layout/Unicorn.lmp6r"

layout_file = open(layout, "r", encoding="shift-jis")
empty_pattern_file = open(empty_pattern, "r", encoding="shift-jis")
layout_reader = csv.reader(layout_file, delimiter="\t")
empty_pattern_reader = csv.reader(empty_pattern_file, delimiter="\t")
layout_list = [row for row in layout_reader]
empty_pattern_list = [row for row in empty_pattern_reader]
layout_file.close()
empty_pattern_file.close()

layout_data = parse_lmp(layout_list)
pattern_data = parse_lmp(empty_pattern_list)

led = layout_data["LAMP"].merge(layout_data["KINDPOSTFIX"], on="Kind")
led_names: list[str] = []

for i, row in led.iterrows():
    led_names.append(row["LabelBase"] + row["Postfix"])

pattern_data["PATTERN"].loc[0] = [0, "BASE", 0, 1000, 0, "", 60, -1, 500, "TRUE", "test pattern", "FALSE", 0]
for i, n in enumerate(led_names):
    pattern_data["LAYER"].loc[i] = [i, 0, n, "", "TRUE", "FALSE"]

output_file = open("./output.lmp6r", "w", encoding="shift-jis", newline="\n")
output_data = export_lmp6r(pattern_data)
output_file.write(output_data)
output_file.close()


