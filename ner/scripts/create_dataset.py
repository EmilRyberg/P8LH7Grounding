import os


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

def get_reverse_mapping(mapping_dict):
    reverse_mapping = {}
    for keyword, entity in mapping_dict.items():
        if entity in reverse_mapping.keys():
            reverse_mapping[entity].append(keyword)
        else:
            reverse_mapping[entity] = [keyword]
    return reverse_mapping

def save_keyword_entity_mapping_to_file(file_path, mapping_dict):
    reverse_mapping = get_reverse_mapping(mapping_dict)
    with open(file_path, "w") as file:
        for entity, keywords in reverse_mapping.items():
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
    keywords_with_multiple_words = [x for x in keyword_to_entity_mapping.keys() if len(x.strip().split(" ")) > 1]
    with open(dataset_path) as f:
        content = f.read().strip()
        lines_splitted = content.splitlines()
        for line in lines_splitted:
            words_splitted = line.split(" ")
            skip_to_index = 0
            for index, word in enumerate(words_splitted):
                if skip_to_index != 0 and index < skip_to_index:
                    continue
                word_to_process = word.replace(".", "")
                has_labelled = False
                for keyword in keywords_with_multiple_words:
                    keyword_splitted = keyword.lower().split(" ")
                    all_match = True
                    for sk_index, sk in enumerate(keyword_splitted):
                        if len(words_splitted) <= index + sk_index:
                            all_match = False
                            break
                        word_splitted_without_period = words_splitted[index + sk_index].replace(".", "")
                        if word_splitted_without_period.lower() != sk:
                            all_match = False
                            break
                    if all_match:
                        has_labelled = True
                        class_name = keyword_to_entity_mapping[keyword]
                        skip_to_index = index + len(keyword_splitted)
                        final_output += f"{word_to_process}\tB-{class_name}\n"
                        for i in range(index + 1, index + len(keyword_splitted)):
                            w_without_period = words_splitted[i].replace(".", "")
                            final_output += f"{w_without_period}\tI-{class_name}\n"
                            if "." in words_splitted[i]:
                                final_output += ".\tO\n"
                        break
                if not has_labelled:
                    if word_to_process.lower() in keyword_to_entity_mapping.keys():
                        class_name = keyword_to_entity_mapping[word_to_process.lower()]
                        final_output += f"{word_to_process}\tB-{class_name}\n"
                    else:
                        final_output += f"{word_to_process}\tO\n"
                if "." in word:
                    final_output += ".\tO\n"
            final_output += "\n"
    print("Specify path to save output dataset:")
    save_path = input("> ")
    with open(save_path, "w") as sf:
        sf.write(final_output)
    print("Create tag file?")
    choice = get_valid_input("[Y/N] ", False, "y", "n")
    if choice == "y":
        print("Write save path for tag file:")
        tag_file_path = input("> ")
        reverse_mapping = get_reverse_mapping(keyword_to_entity_mapping)
        entity_names = reverse_mapping.keys()
        lines = "O\n"
        for en in entity_names:
            lines += f"B-{en}\nI-{en}\n"
        with open(tag_file_path, "w") as tag_file:
            tag_file.write(lines)
    print("Done!")


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
    create_dataset()
