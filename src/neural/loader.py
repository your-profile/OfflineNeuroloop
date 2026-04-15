import pandas as pd
import numpy as np
from src.neural import utils_data as utils

class DataLoader:
    def __init__(self, fnirs_data_source_path=None, task_data_source_path=None, labeled_data_source_path=None, participant_list=None, conditions_list=None):
        self.fnirs_path = fnirs_data_source_path
        self.label_path = labeled_data_source_path
        self.task_path = task_data_source_path
        self.participant_list = participant_list
        self.conditions = conditions_list

    def load_fnirs(self) -> pd.DataFrame:
        """Load fNIRS data."""

        if self.participant_list is None:
            Exception("Participant List not Provided")
        if self.conditions is None:
            Exception("Condition List not Provided")
        if self.fnirs_path is None:
            Exception("fNIRS File Source Path not Provided")

        fnirs_dict = utils.Data.read_files(participant_list=self.participant_list, conditions=self.conditions, source_folder_1=self.fnirs_path)
        df = pd.DataFrame()

        for key in sorted(fnirs_dict.keys()):
            df = pd.concat(
                [df, fnirs_dict[key]],
                ignore_index=True
            )
        # print(df.head(2))
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.sort_values(by=['pid', 'time'])

        return df


    def load_task(self):
        """Load task / RL event data."""

        if self.participant_list is None:
            Exception("Participant List not Provided")
        if self.conditions is None:
            Exception("Condition List not Provided")
        if self.task_path is None:
            Exception("fNIRS File Source Path not Provided")

        task_dict = utils.Data.read_files(participant_list=self.participant_list, conditions=self.conditions, source_folder_1=self.task_path)
        df = self.create_time_dict(task_dict)
        df['time'] = pd.to_datetime(df['dateTimestamps'], utc=True)
        df = df.sort_values(by=['participantKey', 'time'])
        return df


    def load_labels(self):
        """Load event labels."""

        if self.participant_list is None:
            Exception("Participant List not Provided")
        if self.conditions is None:
            Exception("Condition List not Provided")
        if self.label_path is None:
            Exception("fNIRS File Source Path not Provided")

        label_dict = utils.Data.read_files(participant_list=self.participant_list, conditions=self.conditions, source_folder_1=self.label_path)
        
        df = pd.DataFrame()

        for key in sorted(label_dict.keys()):
            df = pd.concat(
                [df, label_dict[key]],
                ignore_index=True
            )

        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.sort_values(by=['pid', 'time'])
        
        return df

    def create_time_dict(self, demo_dict):
        """ create_time_dict(demo_dict: Dict):
            Re-structuring experiment dictionaries into a 2D dataframe.
            Original Dict is (Task Data x Episode x Steps)
            New Dataframe is (Task Data x (Episode x Steps))
        """
        new_dict = {'floatTimestamps':[],
                    'dateTimestamps': [],
                    'participantKey': [],
                    'participant_id': [],
                    'condition': [],
                    'episode': [],
                    'states': [],
                    'actions': [],
                    'rewards':[],
                    'discountedRewards':[],
                    'chosen_actions': [],
                    'optimal_actions': [],
                    'numDemonstrations': [],
                    'steps': [],
                    'seed': [],
        }

        def parse_participant_key(key: str):
            key = str(key)
            participant_str = key[:3]
            cond = key[3:]
            try:
                participant_id = int(participant_str)
            except ValueError:
                participant_id = np.nan
            return participant_id, cond
        
        for p in sorted(demo_dict.keys()):
            if p[3]=="R":
                new_dict['desired_goal'] = []
                new_dict['achieved_goal'] = []

            participant_id, cond = parse_participant_key(p)
            for i in range(len(demo_dict[p].keys())):
                discounted_rewards = self.compute_discounted_rewards(demo_dict[p][i]["rewards"], 0.95)

                for j in range(len(demo_dict[p][i]['actions'])):
                    new_dict['floatTimestamps'].append(demo_dict[p][i]["floatTimestamps"][j])
                    new_dict['dateTimestamps'].append(demo_dict[p][i]["dateTimestamps"][j])
                    new_dict['rewards'].append(demo_dict[p][i]["rewards"][j])
                    new_dict['discountedRewards'].append(discounted_rewards[j])

                    new_dict['chosen_actions'].append(demo_dict[p][i]["chosen_actions"][j])
                    new_dict['optimal_actions'].append(demo_dict[p][i]["optimal_actions"][j])
                
                    new_dict['actions'].append(demo_dict[p][i]["actions"][j])
                    new_dict['episode'].append(demo_dict[p][i]['episode'])
                    new_dict['steps'].append(demo_dict[p][i]['steps'])
                    new_dict['states'].append(demo_dict[p][i]["states"][j])

                    new_dict['numDemonstrations'].append(len(demo_dict[p].keys()))
                    new_dict['participantKey'].append(p)
                    new_dict['participant_id'].append(participant_id)
                    new_dict['condition'].append(cond)
                    new_dict['seed'].append(demo_dict[p][i]["seed"])
                    # print(p)
                    if p[3]=="R":
                        new_dict["desired_goal"].append((demo_dict[p][i]["desired_goal"]))
                        new_dict['achieved_goal'].append(demo_dict[p][i]["achieved_goal"][j])

        # print(demo_dict)
        return pd.DataFrame(new_dict)

    def compute_discounted_rewards(self, rewards, gamma):
        """
        Compute discounted rewards from episode rewards.
        """
        discounted_rewards = np.zeros_like(rewards, dtype=np.float64)
        cumulative_sum = 0.0
        
        for t in (range(len(rewards))):
            if rewards[t] == None:
                continue
            cumulative_sum = rewards[t] + gamma * cumulative_sum
            discounted_rewards[t] = cumulative_sum

        return discounted_rewards