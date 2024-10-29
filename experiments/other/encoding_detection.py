import chardet

with open("data/txt/germany-solar-panels-climate-change_29Jul2024.txt", "rb") as file:
    raw_data = file.read()
    detected_encoding = chardet.detect(raw_data)["encoding"]