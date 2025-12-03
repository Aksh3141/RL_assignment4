
'''
This file contains the configs used for Model creation and training. You need to give your best hyperparameters and the configs you used to get the best results for 
every environment and experiment.  These configs will be automatically loaded and used to create and train your model in our servers.
'''
#You can add extra keys or modify to the values of the existing keys in bottom level of the dictionary.
#DONT CHANGE THE OVERALL STRUCTURE OF THE DICTIONARY. 

configs = {
    
    
    'InvertedPendulum-v4': {
        "PPO":{
            #You can add or change the keys here
                "hyperparameters": {
                    "total_timesteps": 1000000 ,
                    "algorithm": 'PPO',
                    "log_interval" : 500,
                    "eval_freq" : 5000,
                    "save_freq" : 50000,
                    "Run_name" :'Run',
                    
                },            
        },
        
         "A2C":{
            #You can add or change the keys here
                "hyperparameters": {
                    "total_timesteps": 1000000,
                    "algorithm": 'A2C',
                    "log_interval" : 500,
                    "eval_freq" : 5000,
                    "save_freq" : 50000,
                    "Run_name" :'Run',
                    
                },            
        },
    },
    
    'HalfCheetah-v4': {
        
         "PPO":{
            #You can add or change the keys here
                "hyperparameters": {
                    "total_timesteps": 1000000 ,
                    "algorithm": 'PPO',
                    "log_interval" : 500,
                    "eval_freq" : 5000,
                    "save_freq" : 50000,
                    "Run_name" :'Run',
                    
                },            
        },
         "A2C":{
            #You can add or change the keys here
                "hyperparameters": {
                    "total_timesteps": 1000000,
                    "algorithm": 'A2C',
                    "log_interval" : 500,
                    "eval_freq" : 5000,
                    "save_freq" : 50000,
                    "Run_name" : 'Run',
              
                },            
        },
    },
    
    

    'Hopper-v4': {
        
       "PPO":{
            #You can add or change the keys here
                "hyperparameters": {
                    "total_timesteps": 1000000 ,
                    "algorithm": 'PPO',
                    "log_interval" : 500,
                    "eval_freq" : 5000,
                    "save_freq" : 50000,
                    "Run_name" :'Run',
                    
                },            
        },
       "A2C":{
            #You can add or change the keys here
                "hyperparameters": {
                    "total_timesteps": 1000000,
                    "algorithm": 'A2C',
                    "log_interval" : 500,
                    "eval_freq" : 5000,
                    "save_freq" : 50000,
                    "Run_name" : 'Run',
            
                },            
        },
    },
    }
    
    