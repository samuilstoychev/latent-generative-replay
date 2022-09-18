import random 

class ReplayBuffer(): 
    def __init__(self, size, scenario="task"): 
        self.rm = set() 
        self.size = size 
        self.taskId = 1 
        self.scenario = scenario
        self.seen = []

    def add(self, batch): 
        for sample in batch:  
            if len(self.seen) < self.size: 
                self.seen.append(sample)
            else: 
                replace_index = random.randint(0, self.size - 1)
                self.seen[replace_index] = sample
        
    def update(self):
        h = self.size // self.taskId 
        radd = set(random.sample(self.seen, h)) 
        rreplace = set() if self.taskId == 1 else set(random.sample(self.rm, h)) 
        self.rm = (self.rm - rreplace).union(radd)  
        self.taskId += 1
        self.seen = []
        
    def replay(self, batch_size): 
        if not self.rm: 
            return None
        return list(random.sample(self.rm, batch_size))