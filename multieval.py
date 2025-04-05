import multiprocessing as mp
from tqdm import tqdm
#from environment import Game2048Env, TD_MCTS, TD_MCTS_Node,get_approximator # Import your actual modules
from env2048 import Game2048Env
from approximator import NTupleApproximator
from TD_MCTS import TD_MCTS, TD_MCTS_Node
import pickle
def run_experiment(approximator,queue):
    env = Game2048Env()
    state = env.reset()
    
    done = False
    cnt = 0
    while not done:
        td_mcts = TD_MCTS(approximator, iterations=1000, exploration_constant=100, rollout_depth=0)
        root = TD_MCTS_Node()
        root.untried_actions = [a for a in range(4) if env.is_move_legal(a)]
        
        for _ in range(td_mcts.iterations):
            td_mcts.run_simulation(root, env)
            
        best_action, visit_distribution = td_mcts.best_action_distribution(root)
        state, reward, done, _ = env.step(best_action)
        cnt += 1
        root.parent = None

    queue.put((env.score,env.board,cnt))  # Store the score in the queue

# Main parallel execution block
def get_approximator():
    patterns = [
    # straight
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 1), (1, 1), (2, 1), (3, 1)],
    # Square patterns (2x3)
    [(0, 0), (1, 0), (2, 0),
    (0, 1), (1, 1), (2, 1)],
    [(0, 1), (1, 1), (2, 1),
    (0, 2), (1, 2), (2, 2)]
    ]
    approximator = NTupleApproximator(board_size=4,patterns=patterns)
    print('hehe')
    with open('approximator.pkl', 'rb') as f:
        approximator = pickle.load(f)
    
    print("Approximator loaded successfully!")
    return approximator
if __name__ == "__main__":
    num_processes = 10
    processes = []
    mp.set_start_method("spawn")  # Important: Fixes multiprocessing issues on macOS
    
    queue = mp.Queue()  # Use a queue to collect results
    approximator = get_approximator()
    for _ in range(num_processes):
        print('start')
        p = mp.Process(target=run_experiment, args=(approximator,queue,))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # Collect results
    result = [queue.get() for _ in range(num_processes)]
    scores = 0
    min_score = float('inf')
    max_score = float('-inf')
    for score,board,steps in result:
        scores += score
        max_score = max(max_score,score)
        min_score = min(min_score,score)
        print(board,f'score: {score},steps:{steps}',sep='\n')
    print("Average Score:", scores / num_processes)
    print('Max Score:',max_score)
    print('Min Score:',min_score)
