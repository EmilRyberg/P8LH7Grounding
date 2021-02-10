
colors = [
    "yellow",
    "black",
    "blue",
    "white",
    "brown",
    "green"
]

objects = [
    "bottom",
    "cover",
    "fuse"
]

CLASS_NONE = 0
CLASS_O = 1
CLASS_BColor = 2
CLASS_IColor = 3
CLASS_BObject = 4
CLASS_IObject = 5

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
                elif word_to_process in objects:
                    if word_class_before == CLASS_BObject or word_class_before == CLASS_IObject:
                        final_output += f"{word_to_process}\tI-object\n"
                        word_class_before = CLASS_IObject
                    else:
                        final_output += f"{word_to_process}\tB-object\n"
                        word_class_before = CLASS_BObject
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
