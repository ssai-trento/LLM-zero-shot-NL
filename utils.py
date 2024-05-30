import os


def create_dir(dir):
    # if dir does not exist, create it
    if not os.path.exists(dir):
        os.makedirs(dir)


def int_to_days(int_day):
    days_of_week = {0: 'Monday',
                    1: 'Tuesday',
                    2: 'Wednesday',
                    3: 'Thursday',
                    4: 'Friday',
                    5: 'Saturday',
                    6: 'Sunday'}
    return days_of_week.get(int_day, "NA")


def list_predicted_users(folder_path):
    # get the names of all the files in the folder
    files = os.listdir(folder_path)
    # filter out only the files that are .json
    files = [f for f in files if f.endswith('.json')]
    # split file names to get the user id (second last _ is the split)
    users = [f.split('_')[-2] for f in files]
    # remove duplicates
    users = list(set(users))
    return users

def prompt_generator(v, prompt_type):
    prompt = ''
    if prompt_type == '1_chat_llama':
        prompt = f"""
                [INST] <<SYS>>
                You are designed to predict the next locate of the users.
                <</SYS>>
    		Your task is to predict <next_place_id> in <target_stay>, a location with an unknown ID, while temporal data is available.

                Predict <next_place_id> by considering:
                1. The user's activity trends gleaned from <historical_stays> and the current activities from  <context_stays>.
                2. Temporal details (start_time and day_of_week) of the target stay, crucial for understanding activity variations.

                Present your answer in a JSON object with:
                "prediction" (IDs of the five most probable places, ranked by probability) and "reason" (a concise justification for your prediction).
                
		The data:
                    <historical_stays>: {v['historical_stays']}
                    <context_stays>: {v['context_stays']}
                    <target_stay>: {v['target_stay']}
                    [/INST]"""
    elif prompt_type == '1':
        prompt = f"""
    		Your task is to predict <next_place_id> in <target_stay>, a location with an unknown ID, while temporal data is available.

                Predict <next_place_id> by considering:
                1. The user's activity trends gleaned from <historical_stays> and the current activities from  <context_stays>.
                2. Temporal details (start_time and day_of_week) of the target stay, crucial for understanding activity variations.

                Present your answer in a JSON object with:
                "prediction" (IDs of the five most probable places, ranked by probability) and "reason" (a concise justification for your prediction).
                
		The data:
                    <historical_stays>: {v['historical_stays']}
                    <context_stays>: {v['context_stays']}
                    <target_stay>: {v['target_stay']}
                   """
    elif prompt_type == '2':
        prompt = f"""
        original prompt: 
    		Your task is to predict <next_place_id> in <target_stay>, a location with an unknown ID, while temporal data is available.

                Predict <next_place_id> by considering:
                1. The user's activity trends gleaned from <historical_stays> and the current activities from  <context_stays>.
                2. Temporal details (start_time and day_of_week) of the target stay, crucial for understanding activity variations.

                Present your answer in a JSON object with:
                "prediction" (IDs of the five most probable places, ranked by probability) and "reason" (a concise justification for your prediction).
                
		The data:  
                        historical_stays: [['10:49 PM',
                            'Sunday',
                            '4b80bafef964a520ee8830e3'],
                            ['03:16 AM', 'Monday', '4c182e2c6a21c9b6f2bbc897'],
                            ['08:57 AM', 'Monday', '4b8ef710f964a5209c4133e3'],
                            ['09:02 AM', 'Monday', '4c54e77172cf0f47229246d5'],
                            ['12:06 PM', 'Monday', '4dff7dd51495f702193690bf'],
                            ['12:12 PM', 'Monday', '4b7e3467f964a520fde52fe3'],
                            ['01:54 PM', 'Monday', '4b22e836f964a520185024e3'],
                            ['02:46 PM', 'Monday', '4b80bafef964a520ee8830e3'],
                            ['03:13 PM', 'Wednesday', '4b80bafef964a520ee8830e3'],
                            ['02:10 AM', 'Thursday', '4bb365f54019a593e6d937b8'],
                            ['11:29 PM', 'Sunday', '4b80bafef964a520ee8830e3'],
                            ['12:32 AM', 'Monday', '4bb365f54019a593e6d937b8'],
                            ['12:59 AM', 'Monday', '4b7e3467f964a520fde52fe3'],
                            ['02:22 PM', 'Thursday', '4b5d54a2f964a5200e5a29e3'],
                            ['01:40 PM', 'Friday', '4cdbe66e22bd721e4302f847'],
                            ['02:03 PM', 'Saturday', '4e13debbe4cd473c968b5afc'],
                            ['12:17 AM', 'Friday', '4bf5d425004ed13aa27541a0'],
                            ['11:04 AM', 'Friday', '4c2c8fd677cfe21e2029b6f1'],
                            ['12:17 AM', 'Monday', '4bf5d425004ed13aa27541a0'],
                            ['01:09 AM', 'Tuesday', '4b558306f964a5201be627e3'],
                            ['10:34 AM', 'Tuesday', '4dabc19f5da3ba8a47a875cd'],
                            ['10:44 AM', 'Wednesday', '4bf5d425004ed13aa27541a0'],
                            ['10:51 AM', 'Wednesday', '4bea97ff9fa3ef3b5d2680c9'],
                            ['08:58 AM', 'Thursday', '4b7e3467f964a520fde52fe3'],
                            ['07:26 AM', 'Friday', '4dff7dd51495f702193690bf'],
                            ['07:32 AM', 'Friday', '4bb365f54019a593e6d937b8'],
                            ['07:33 AM', 'Friday', '4b7e3467f964a520fde52fe3'],
                            ['12:19 AM', 'Tuesday', '4bf5d425004ed13aa27541a0']],
                        context_stays: [['12:09 AM', 'Wednesday', '4dff7dd51495f702193690bf'],
                            ['10:44 AM', 'Wednesday', '4b8da54ef964a5202e0633e3'],
                            ['11:04 AM', 'Wednesday', '4dff7dd51495f702193690bf'],
                            ['11:12 AM', 'Wednesday', '4b7e3467f964a520fde52fe3'],
                            ['02:24 PM', 'Wednesday', '4b22e836f964a520185024e3']],
                        target_stay: ['02:36 AM', 'Thursday', '<next_place_id>']

                        prediction:   ['4b80bafef964a520ee8830e3',
                            '4b558306f964a5201be627e3',
                            '4b7e3467f964a520fde52fe3',
                            '4c182e2c6a21c9b6f2bbc897',
                            '4dff7dd51495f702193690bf'],
                        'reason': 'User has repeatedly visited places in the evening and at night, especially on weekdays. The most recent context stays are also at nighttime.'

        original prompt:
    		Your task is to predict <next_place_id> in <target_stay>, a location with an unknown ID, while temporal data is available.

                Predict <next_place_id> by considering:
                1. The user's activity trends gleaned from <historical_stays> and the current activities from  <context_stays>.
                2. Temporal details (start_time and day_of_week) of the target stay, crucial for understanding activity variations.

                Present your answer in a JSON object with:
                "prediction" (IDs of the five most probable places, ranked by probability) and "reason" (a concise justification for your prediction).
                
		The data:
                    <historical_stays>: {v['historical_stays']}
                    <context_stays>: {v['context_stays']}
                    <target_stay>: {v['target_stay']}
                    """
    elif prompt_type == '3':
        prompt = f"""
                        original prompt: {{
                            historical_stays: [['10:49 PM','Sunday','4b80bafef964a520ee8830e3'],
                                ['03:16 AM', 'Monday', '4c182e2c6a21c9b6f2bbc897'],
                                ['08:57 AM', 'Monday', '4b8ef710f964a5209c4133e3'],
                                ['09:02 AM', 'Monday', '4c54e77172cf0f47229246d5'],
                                ['12:06 PM', 'Monday', '4dff7dd51495f702193690bf'],
                                ['12:12 PM', 'Monday', '4b7e3467f964a520fde52fe3'],
                                ['01:54 PM', 'Monday', '4b22e836f964a520185024e3'],
                                ['02:46 PM', 'Monday', '4b80bafef964a520ee8830e3'],
                                ['03:13 PM', 'Wednesday', '4b80bafef964a520ee8830e3'],
                                ['02:10 AM', 'Thursday', '4bb365f54019a593e6d937b8'],
                                ['11:29 PM', 'Sunday', '4b80bafef964a520ee8830e3'],
                                ['12:32 AM', 'Monday', '4bb365f54019a593e6d937b8'],
                                ['12:59 AM', 'Monday', '4b7e3467f964a520fde52fe3'],
                                ['02:22 PM', 'Thursday', '4b5d54a2f964a5200e5a29e3'],
                                ['01:40 PM', 'Friday', '4cdbe66e22bd721e4302f847'],
                                ['02:03 PM', 'Saturday', '4e13debbe4cd473c968b5afc'],
                                ['12:17 AM', 'Friday', '4bf5d425004ed13aa27541a0'],
                                ['11:04 AM', 'Friday', '4c2c8fd677cfe21e2029b6f1'],
                                ['12:17 AM', 'Monday', '4bf5d425004ed13aa27541a0'],
                                ['01:09 AM', 'Tuesday', '4b558306f964a5201be627e3'],
                                ['10:34 AM', 'Tuesday', '4dabc19f5da3ba8a47a875cd'],
                                ['10:44 AM', 'Wednesday', '4bf5d425004ed13aa27541a0'],
                                ['10:51 AM', 'Wednesday', '4bea97ff9fa3ef3b5d2680c9'],
                                ['08:58 AM', 'Thursday', '4b7e3467f964a520fde52fe3'],
                                ['07:26 AM', 'Friday', '4dff7dd51495f702193690bf'],
                                ['07:32 AM', 'Friday', '4bb365f54019a593e6d937b8'],
                                ['07:33 AM', 'Friday', '4b7e3467f964a520fde52fe3'],
                                ['12:19 AM', 'Tuesday', '4bf5d425004ed13aa27541a0']],
                            context_stays: [['12:09 AM', 'Wednesday', '4dff7dd51495f702193690bf'],
                                ['10:44 AM', 'Wednesday', '4b8da54ef964a5202e0633e3'],
                                ['11:04 AM', 'Wednesday', '4dff7dd51495f702193690bf'],
                                ['11:12 AM', 'Wednesday', '4b7e3467f964a520fde52fe3'],
                                ['02:24 PM', 'Wednesday', '4b22e836f964a520185024e3']]}}

                            target_stay:   {{'prediction': ['4b80bafef964a520ee8830e3',
                                '4b558306f964a5201be627e3',
                                '4b7e3467f964a520fde52fe3',
                                '4c182e2c6a21c9b6f2bbc897',
                                '4dff7dd51495f702193690bf'],
                                'reason': 'User has repeatedly visited places in the evening and at night, especially on weekdays. The most recent context stays are also at nighttime.']}}

                        original prompt:
                            <historical_stays>: {v['historical_stays']}
                            <context_stays>: {v['context_stays']}
                            <target_stay>: {v['target_stay']}
                    """
    elif prompt_type == '5':
        prompt = f"""
   		Your task is to predict <next_place_id> in <target_stay>, a location with an unknown ID, while temporal data is available.

                Predict <next_place_id> by considering:
                1. The user's activity trends gleaned from <historical_stays> and the current activities from  <context_stays>.
                2. Temporal details (start_time and day_of_week) of the target stay, crucial for understanding activity variations.

                Present your answer in a JSON object with:
                "prediction" (IDs of the five most probable places, ranked by probability) and "reason" (a concise justification for your prediction).
            
                    Consider the following examples to understand the user's patterns:
            
                    1.
                    historical_stays: [['10:49 PM','Sunday','4b80bafef964a520ee8830e3'],
                                            ['03:16 AM', 'Monday', '4c182e2c6a21c9b6f2bbc897'],
                                            ['08:57 AM', 'Monday', '4b8ef710f964a5209c4133e3'],
                                            ['02:46 PM', 'Monday', '4b80bafef964a520ee8830e3'],
                                            ['03:13 PM', 'Wednesday', '4b80bafef964a520ee8830e3'],
                                            ['02:10 AM', 'Thursday', '4bb365f54019a593e6d937b8'],
                                            ['11:29 PM', 'Sunday', '4b80bafef964a520ee8830e3'],
                                            ['12:32 AM', 'Monday', '4bb365f54019a593e6d937b8'],
                                            ['12:59 AM', 'Monday', '4b7e3467f964a520fde52fe3'],
                                            ['02:22 PM', 'Thursday', '4b5d54a2f964a5200e5a29e3'],
                                            ['01:40 PM', 'Friday', '4cdbe66e22bd721e4302f847'],
                                            ['02:03 PM', 'Saturday', '4e13debbe4cd473c968b5afc'],
                                            ['12:17 AM', 'Friday', '4bf5d425004ed13aa27541a0'],
                                            ['11:04 AM', 'Friday', '4c2c8fd677cfe21e2029b6f1'],
                                            ['08:58 AM', 'Thursday', '4b7e3467f964a520fde52fe3'],
                                            ['07:26 AM', 'Friday', '4dff7dd51495f702193690bf'],
                                            ['07:32 AM', 'Friday', '4bb365f54019a593e6d937b8'],
                                            ['07:33 AM', 'Friday', '4b7e3467f964a520fde52fe3'],
                                            ['12:19 AM', 'Tuesday', '4bf5d425004ed13aa27541a0']],
                    context_stays: [['12:09 AM', 'Wednesday', '4dff7dd51495f702193690bf'],
                                            ['10:44 AM', 'Wednesday', '4b8da54ef964a5202e0633e3'],
                                            ['11:04 AM', 'Wednesday', '4dff7dd51495f702193690bf'],
                                            ['11:12 AM', 'Wednesday', '4b7e3467f964a520fde52fe3'],
                                            ['02:24 PM', 'Wednesday', '4b22e836f964a520185024e3']]
            
                    target_stay:   {{'prediction': ['4b80bafef964a520ee8830e3',
                                            '4b558306f964a5201be627e3',
                                            '4b7e3467f964a520fde52fe3',
                                            '4c182e2c6a21c9b6f2bbc897',
                                            '4dff7dd51495f702193690bf']}}
            
            
                    2.
                    historical_stays: ['03:47 AM', 'Monday', '4b7e3467f964a520fde52fe3'],
                                        ['10:45 PM', 'Tuesday', '4b80bafef964a520ee8830e3'],
                                        ['11:39 PM', 'Tuesday', '4dff7dd51495f702193690bf'],
                                        ['11:45 PM', 'Tuesday', '4bb365f54019a593e6d937b8'],
                                        ['03:01 PM', 'Thursday', '4b22e836f964a520185024e3']
                    context_stays: [[['12:09 AM', 'Wednesday', '4dff7dd51495f702193690bf'],
                                        ['10:44 AM', 'Wednesday', '4b8da54ef964a5202e0633e3'],
                                        ['11:04 AM', 'Wednesday', '4dff7dd51495f702193690bf'],
                                        ['11:12 AM', 'Wednesday', '4b7e3467f964a520fde52fe3'],
                                        ['02:24 PM', 'Wednesday', '4b22e836f964a520185024e3']]
                    target_stay:   {{'prediction': ['4b6ba709f964a52059142ce3',
                    '4c182e2c6a21c9b6f2bbc897',
                        '4b8ef710f964a5209c4133e3',
                        '4bea89ca415e20a1af16e5bb',
                        '4dff7dd51495f702193690bf']}}
            
            
		Predict <next_place_id> by considering:
                1. The user's activity trends gleaned from <historical_stays> and the current activities from  <context_stays>.
                2. Temporal details (start_time and day_of_week) of the target stay, crucial for understanding activity variations.

                Present your answer in a JSON object with:
                "prediction" (IDs of the five most probable places, ranked by probability) and "reason" (a concise justification for your prediction).
            
                    The data are as follows:
                    <historical_stays>: {v['historical_stays']}
                    <context_stays>: {v['context_stays']}
                    <target_stay>: {v['target_stay']}
                    """
    return prompt
