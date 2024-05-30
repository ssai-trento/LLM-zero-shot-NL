import utils
from data import Dataset
from llms import LLM


def run():
    dataset = Dataset(dataset_name='nyc', trajectory_mode='trajectory_split', historical_stays=15,
                      context_stays=6, save_dir='data/processed')

    test_dictionary, true_locations = dataset.get_generated_datasets()
    
    # Select the model and the prompt to run

    print('Running 13 B')
    llm = LLM(test_dictionary, true_locations, prompt_type='1',
              model_name='llama13bchat', trajectory_mode='trajectory_split')

    # OTHER EXAMPLES

    #print('Running 7 B')
    #llm = LLM(test_dictionary, true_locations, prompt_type='5',
     #        model_name='llama7bchat', trajectory_mode='trajectory_split')

    #print('Running 70 B')
    #llm = LLM(test_dictionary, true_locations, prompt_type='5',
             # model_name='llama70bchat', trajectory_mode='trajectory_split')

    
    #print('Running gpt 3.5 turbo')
    #llm = LLM(test_dictionary, true_locations, prompt_type='5',
     #        model_name='gpt35turbo', trajectory_mode='trajectory_split')
            

if __name__ == '__main__':
    run()
