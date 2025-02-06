from data.task import Task
from collections import Counter

def top_k(candidate_space, scores, k=1):
    """
    Returns the top-k factors based on scores.
    """
    sorted_factors = sorted(zip(candidate_space, scores), key=lambda x: x[1], reverse=True)
    return [factor for factor, _ in sorted_factors[:k]]

class Decompiling:

    def __init__(self, is_uniform=True, is_brute_force=True, factor_size=4):
        self.is_uniform = is_uniform
        self.is_brute_force = is_brute_force
        self.factor_size = factor_size

        if self.is_uniform:
            self.do = self.do_uniform

        if self.is_brute_force:
            self.candidate_space = self.compute_whole_candidate_space

    def compute_whole_candidate_space(self, session: list[Task]):
        candidate_factors = []
        for task in session:
            if task.is_solved:
                solution = task.solution.trajectory()
                offset = 0
                while offset + self.factor_size <= len(solution):
                    factor = solution[offset:offset+self.factor_size]
                    if factor not in candidate_factors:  # Ensure uniqueness
                        candidate_factors.append(factor)
                    offset += 1
        
        return candidate_factors

    def do_uniform(self, session: list[Task], k=1, vocabulary_size=4):
        candidate_space = self.compute_whole_candidate_space(session)
        programs = [task.solution.trajectory() for task in session if task.is_solved]
        weight_per_program = [(len(rho) * (vocabulary_size**len(rho)))**-1 for rho in programs]

        scores = []
        for f in candidate_space:
            score = 0
            for r, rho in enumerate(programs):
                factor_freq = sum(1 for i in range(len(rho) - self.factor_size + 1) if rho[i:i+self.factor_size] == f)
                score += weight_per_program[r] * factor_freq

            score *= len(f)
            scores.append(score)

        factors = top_k(candidate_space, scores, k=k)
        return factors
