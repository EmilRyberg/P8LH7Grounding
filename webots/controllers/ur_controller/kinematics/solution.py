class InverseKinematicsSpecificSolution:
    def __init__(self):
        self.is_valid_solution = True
        self.thetas = []


class InverseKinematicsShoulderSolution:
    def __init__(self):
        self.is_valid_solution = True
        self.solution_elbow_up = InverseKinematicsSpecificSolution()
        self.solution_elbow_down = InverseKinematicsSpecificSolution()


class InverseKinematicsWristSolution:
    def __init__(self):
        self.is_valid_solution = True
        self.solution_wrist_up = InverseKinematicsShoulderSolution()
        self.solution_wrist_down = InverseKinematicsShoulderSolution()


class InverseKinematicsSolution:
    def __init__(self):
        self.solution_shoulder_left = InverseKinematicsWristSolution()
        self.solution_shoulder_right = InverseKinematicsWristSolution()
