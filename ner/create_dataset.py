
colors = [
    "yellow",
    "black",
    "blue",
    "white",
    "brown",
    "green"
]

locations = [
    "left",
    "right",
    "above",
    "below",
    "middle",
    "on",
    "top"
]

objects = [
    "bottom",
    "cover",
    "fuse",
    "box",
    "table"
]

grasp_actions = [
    "pick",
    "up",
    "grasp",
    "take",
    "Pick",
    "Grasp",
    "Take"
]

find_actions = [
    "find",
    "locate",
    "where",
    "Find",
    "Locate",
    "Where"
]

CLASS_NONE = 0
CLASS_O = 1
CLASS_BColor = 2
CLASS_IColor = 3
CLASS_BObject = 4
CLASS_IObject = 5
CLASS_BGrasp = 6
CLASS_IGrasp = 7
CLASS_BFind = 8
CLASS_IFind = 9
CLASS_BLocation = 10
CLASS_ILocation = 11

def create_dataset(file_path, save_path):
    final_output = ""
    with open(file_path) as f:
        content = f.read().strip()
        lines_splitted = content.splitlines()
        for line in lines_splitted:
            words_splitted = line.split(" ")
            word_class_before = CLASS_O
            for word in words_splitted:
                word_to_process = word.replace(".", "")
                if word_to_process in colors:
                    final_output += f"{word_to_process}\tB-color\n"
                    word_class_before = CLASS_BColor
                elif word_to_process in locations:
                    if word_class_before == CLASS_BLocation or word_class_before == CLASS_ILocation:
                        final_output += f"{word_to_process}\tI-location\n"
                        word_class_before = CLASS_ILocation
                    else:
                        final_output += f"{word_to_process}\tB-location\n"
                        word_class_before = CLASS_BLocation
                elif word_to_process in objects:
                    if word_class_before == CLASS_BObject or word_class_before == CLASS_IObject:
                        final_output += f"{word_to_process}\tI-object\n"
                        word_class_before = CLASS_IObject
                    else:
                        final_output += f"{word_to_process}\tB-object\n"
                        word_class_before = CLASS_BObject
                elif word_to_process in grasp_actions:
                    if word_class_before == CLASS_BGrasp or word_class_before == CLASS_IGrasp:
                        final_output += f"{word_to_process}\tI-grasp\n"
                        word_class_before = CLASS_IGrasp
                    else:
                        final_output += f"{word_to_process}\tB-grasp\n"
                        word_class_before = CLASS_BGrasp
                elif word_to_process in find_actions:
                    if word_class_before == CLASS_BFind or word_class_before == CLASS_IFind:
                        final_output += f"{word_to_process}\tI-find\n"
                        word_class_before = CLASS_IFind
                    else:
                        final_output += f"{word_to_process}\tB-find\n"
                        word_class_before = CLASS_BFind
                else:
                    final_output += f"{word_to_process}\tO\n"
                    word_class_before = CLASS_O
                if "." in word:
                    final_output += ".\tO\n"
            final_output += "\n"
    with open(save_path, "w") as sf:
        sf.write(final_output)


if __name__ == "__main__":
    create_dataset("dataset_2.txt", "output.txt")
