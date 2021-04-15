class BaseTask:
    def __init__(self):
        self.child_tasks = []

    def __str__(self):
        if len(self.child_tasks) == 0:
            return ""
        output_str = f"Child tasks:\n"
        for task in self.child_tasks:
            output_str += f"\t{task}"
        return output_str


class PickUpTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.object_to_pick_up = None

    def __str__(self):
        return f"Task type: {PickUpTask.__name__}\n\tObject to pick up: {self.object_to_pick_up}\n{super().__str__()}"


class MoveTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.object_to_move = None

    def __str__(self):
        return f"Task type: {MoveTask.__name__}\n\tObject to find: {self.object_to_move}\n{super().__str__()}"


class PlaceTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.object_to_pick_up = None

    def __str__(self):
        return f"Task type: {PickUpTask.__name__}\n\tObject to pick up: {self.object_to_pick_up}\n{super().__str__()}"


class FindTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.object_to_find = None

    def __str__(self):
        return f"Task type: {FindTask.__name__}\n\tObject to find: {self.object_to_find}\n{super().__str__()}"