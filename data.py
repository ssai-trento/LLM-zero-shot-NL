import pandas as pd
import numpy as np
import os
import requests
import tqdm
from sklearn.model_selection import train_test_split
import json

import utils


class Dataset:

    def __init__(self, dataset_name='nyc', trajectory_mode='user_split', historical_stays=40, context_stays=6,
                 hour_bins=72, traj_min_len=5, save_dir='data/processed'):
        self.dataset_url = 'http://www-public.tem-tsp.eu/~zhang_da/pub/dataset_tsmc2014.zip'

        self.dataset_name = dataset_name
        self.trajectory_mode = trajectory_mode
        self.history_stays = historical_stays
        self.context_stays = context_stays
        self.save_dir = save_dir
        self.hour_bins = hour_bins
        self.traj_min_len = traj_min_len

        self.test_dictionary = {}
        self.true_locations = {}

        self.test_size = 0.2
        self.seed = 1234

        # check dataset name
        if self.dataset_name not in ['nyc', 'tky']:
            raise ValueError('Invalid dataset name! Please use either nyc or tky')

        if self.trajectory_mode not in ['user_split', 'trajectory_split']:
            raise ValueError('Invalid trajectory mode! Please use either user_split or trajectory_split')

        # download the dataset if necessary
        self.download_dataset()

        print('Loading the dataset...')
        self.data = self.get_dataset()

        print('Computing trajectories...')
        self.get_trajectories()

    def get_generated_datasets(self):
        return self.test_dictionary, self.true_locations

    def download_dataset(self):
        """
        A function to download the dataset from the URL if the dataset is not present in the current directory
        :return: None
        """

        # check if the dataset is already present in the current directory
        if not os.path.exists('data/dataset_tsmc2014/dataset_TSMC2014_NYC.txt'):
            # download the dataset
            print('Downloading the dataset...')
            r = requests.get(self.dataset_url)
            with open('data/dataset_tsmc2014.zip', 'wb') as f:
                f.write(r.content)
            print('Download complete!')

            # extract the content of the zip folder
            print('Extracting the zip folder...')
            os.system('unzip data/dataset_tsmc2014.zip -d data/')
            print('Extraction complete!')

            # remove the zip folder
            print('Removing the zip folder...')
            os.system('rm data/dataset_tsmc2014.zip')
            print('Removal complete!')

        else:
            print('Dataset already present in the current directory!')

    def get_dataset(self):
        """
        A function to load the dataset into a pandas dataframe
        :return:
        """
        if self.dataset_name == 'nyc':
            data = pd.read_csv('data/dataset_tsmc2014/dataset_TSMC2014_NYC.txt', delimiter="\t", header=None,
                                   names=["user_id", "venue_id", "venue_category", "venue_category_name", "latitude",
                                          "longitude", "timezone_offset", "utc_time"], encoding='ISO-8859-1')
        else:
            data = pd.read_csv('data/dataset_tsmc2014/dataset_TSMC2014_TKY.txt', delimiter="\t", header=None,
                               names=["user_id", "venue_id", "venue_category", "venue_category_name", "latitude",
                                      "longitude", "timezone_offset", "utc_time"], encoding='ISO-8859-1')

        # convert date time to pandas datetime
        data['datetime'] = pd.to_datetime(data['utc_time'])
        # create a column for the day of the week
        data['weekday'] = data['datetime'].dt.dayofweek
        # convert the weekday to a string
        data['weekday'] = data['weekday'].apply(lambda x: utils.int_to_days(x))
        # create a column for the hour of the day
        data['hour'] = data['datetime'].dt.hour
        # create a column with 'AM' if the hour is less than 12, else 'PM'
        data['am_pm'] = np.where(data['hour'] < 12, 'AM', 'PM')
        # if column 'hour' is higher than 12, subtract 12 from it
        data['hour'] = np.where(data['hour'] > 12, data['hour'] - 12, data['hour'])
        # concatenate the hour and am_pm columns
        data['hour'] = data['hour'].astype(str) + ' ' + data['am_pm']

        return data

    def get_trajectories(self):
        # order the data by user_id and datetime
        self.data = self.data.sort_values(by=['user_id', 'datetime'])

        if self.trajectory_mode == 'user_split':
            # get the unique user ids
            unique_user_ids = self.data['user_id'].unique()
            train_users, test_users = train_test_split(unique_user_ids,
                                                       test_size=self.test_size,
                                                       random_state=self.seed)
            train_data = self.data[self.data['user_id'].isin(train_users)]
            test_data = self.data[self.data['user_id'].isin(test_users)]

            # for each user in test data, we retrieve the historical stays (iloc[-(M+N):-N]) and the context stays (iloc[-N:])
            print('Processing training and test split using method user_split...')
            for user_id in tqdm.tqdm(test_data['user_id'].unique()):
                user_data = test_data[test_data['user_id'] == user_id]
                historical_data = user_data.iloc[-(self.history_stays + self.context_stays):-self.context_stays][['hour', 'weekday', 'venue_id']].values
                context_data = user_data.iloc[-self.context_stays:][['hour', 'weekday', 'venue_id']].values
                ground_truth = context_data[-1][-1]
                target_data = context_data[-1][:-1].tolist()
                target_data.append('<next_place_id>')
                context_data = context_data[:-1]

                user_data_dict = {
                    'historical_stays': historical_data.tolist(),
                    'context_stays': context_data.tolist(),
                    'target_stay': target_data,
                }

                self.test_dictionary[str(user_id)] = user_data_dict
                self.true_locations[str(user_id)] = ground_truth

        elif self.trajectory_mode == 'trajectory_split':

            def _group_by_time_window(data):
                # Sort by timestamp
                data = data.sort_values('datetime')
                # Start time for the first group
                start_time = data.iloc[0]['datetime']
                # Initialize trajectory ID

                trajectory_id = self.data['traj_id'].max()
                time_window = pd.Timedelta(hours=self.hour_bins)
                for i in range(1, len(data)):
                    # If the current timestamp is outside the 72-hour window, increment the trajectory ID
                    if data.iloc[i]['datetime'] > start_time + time_window:
                        trajectory_id += 1
                        start_time = data.iloc[i]['datetime']
                    data.iloc[i, data.columns.get_loc('traj_id')] = trajectory_id

                return data

            # Applying the function to group by ID and then by time window
            self.data['traj_id'] = 0
            self.data = self.data.groupby('user_id').apply(_group_by_time_window).reset_index(drop=True)

            # For each user, we retrieve the list of trajectory ids:
            # - the first 80% of the trajectory ids are used for training
            # - the last 20% of the trajectory ids are used for testing
            print('Processing training and test split using method trajectory_split...')
            for user_id in tqdm.tqdm(self.data['user_id'].unique()):
                # put the user in the dictionaries
                self.test_dictionary[str(user_id)] = {}
                self.true_locations[str(user_id)] = {}

                # exclude users with less than 5 trajectories id
                if len(self.data[self.data['user_id'] == user_id]['traj_id'].unique()) < self.traj_min_len:
                    continue

                trajectory_ids = self.data[self.data['user_id'] == user_id]['traj_id'].unique()
                train_trajectory_ids = trajectory_ids[:int(0.8 * len(trajectory_ids))]
                test_trajectory_ids = trajectory_ids[int(0.8 * len(trajectory_ids)):]

                # for each trajectory id in test trajectory ids, we retrieve the historical stays
                # (location ids and time from training) and the context stays (location ids and time
                # from testing with specific trajectory id)

                for trajectory_id in test_trajectory_ids:
                    # exclude this trajectory if it has less than 4 stays
                    if len(self.data[(self.data['user_id'] == user_id) & (self.data['traj_id'] == trajectory_id)]) < 4:
                        continue

                    historical_data = self.data[(self.data['user_id'] == user_id) &
                                                (self.data['traj_id'].isin(train_trajectory_ids))][['hour', 'weekday', 'venue_id']].values
                    context_data = self.data[(self.data['user_id'] == user_id) &
                                             (self.data['traj_id'] == trajectory_id)][['hour', 'weekday', 'venue_id']].values
                    ground_truth = context_data[-1][-1]
                    target_data = context_data[-1][:-1].tolist()
                    target_data.append('<next_place_id>')
                    context_data = context_data[:-1]

                    user_data_dict = {
                        'historical_stays': historical_data.tolist()[-self.history_stays:],
                        'context_stays': context_data.tolist(),
                        'target_stay': target_data,
                    }

                    self.test_dictionary[str(user_id)][str(trajectory_id)] = user_data_dict
                    self.true_locations[str(user_id)][str(trajectory_id)] = ground_truth

        # save the test dictionary and the true locations dictionary
        utils.create_dir(self.save_dir)
        with open(os.path.join(self.save_dir, 'test_dictionary_'+self.dataset_name+'_'+self.trajectory_mode+'.json'), 'w') as f:
            json.dump(self.test_dictionary, f)
        with open(os.path.join(self.save_dir, 'true_locations_'+self.dataset_name+'_'+self.trajectory_mode+'.json'), 'w') as f:
            json.dump(self.true_locations, f)
