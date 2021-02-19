import os
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

def create_dataset_old(file_path, save_path):
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


def parse_keywords_to_entities(file_path):
    with open(file_path, "r") as file:
        content = file.read().strip()
    output_dict = {}
    lines = content.splitlines()
    for line in lines:
        entries = line.split(",")
        entity_name = entries[0]
        keywords = entries[1:]
        for keyword in keywords:
            output_dict[keyword] = entity_name
    return output_dict


def save_keyword_entity_mapping_to_file(file_path, mapping_dict):
    reverse_mapping = {}
    for keyword, entity in mapping_dict.items():
        if entity in reverse_mapping:
            reverse_mapping[entity].append(keyword)
        else:
            reverse_mapping[entity] = [keyword]
    with open(file_path, "w") as file:
        for entity, keywords in reverse_mapping:
            keyword_string = ",".join(keywords)
            line = f"{entity},{keyword_string}\n"
            file.write(line)


def create_dataset():
    keyword_to_entity_mapping = {}
    print("Load file with keywords to entity mapping or create new one? Y=Load File, N=Create New")
    choice = get_valid_input("[Y/N] ", False, "y", "n")
    if choice == "y":
        print("Specify file path")
        file_path = get_valid_file_input()
        keyword_to_entity_mapping = parse_keywords_to_entities(file_path)
    else:
        while True:
            print("Write entity name (without I or B prefix) or press enter to finish:")
            entity_name = input("> ").strip()
            if entity_name == "" or entity_name == "\n":
                break
            print("Write keywords for entity (can be multiple words, such as 'bottom cover'), press enter to stop:")
            while True:
                keyword = input("> ").strip()
                if keyword == "" or keyword == "\n":
                    break
                else:
                    keyword_to_entity_mapping[keyword] = entity_name
        print("Save to file?")
        choice = get_valid_input("[Y/N] ", False, "y", "n")
        if choice == "y":
            print("Specify filename (without extension):")
            file_name = input("> ")
            save_keyword_entity_mapping_to_file(f"{file_name}.txt", keyword_to_entity_mapping)

    print("Specify dataset path:")
    dataset_path = get_valid_file_input()
    final_output = ""
    with open(dataset_path) as f:
        content = f.read().strip()
        lines_splitted = content.splitlines()
        for line in lines_splitted:
            words_splitted = line.split(" ")
            word_class_before = CLASS_O
            for word in words_splitted:
                word_to_process = word.replace(".", "")
            final_output += "\n"
    with open(save_path, "w") as sf:
        sf.write(final_output)


def get_valid_input(prompt, case_sensitive, *valid_inputs):
    while True:
        user_input = input(prompt)
        if not case_sensitive:
            if user_input.lower() in valid_inputs:
                return user_input.lower()
        else:
            if user_input in valid_inputs:
                return user_input
        print("Invalid value")


def get_valid_file_input():
    while True:
        user_input = input("> ")
        if os.path.isfile(user_input):
            return user_input
        print(f"'{user_input}' is not a file.")


if __name__ == "__main__":
    create_dataset("dataset_2.txt", "output.txt")
