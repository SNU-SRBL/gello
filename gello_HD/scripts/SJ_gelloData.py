import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

data_directory = '../rawdata'
target_data_directory = '0130_160514'

class SJ_GelloDataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._data = []
        self._data_loader()
        self._time_order()
    
    def __del__(self):
        pass

    def _data_loader(self):
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.pkl'):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, 'rb') as file:
                    dataframe = self._data_parser(filename, file)
                self._data.append(dataframe)  
    
    def _data_parser(self, filename, file):
        dataframe = {}
        timestr = filename.removesuffix('.pkl')
        timestr = timestr.replace('_', ':')
        time = datetime.fromisoformat(timestr)
        dataframe['time'] = time
        raw_data = pickle.load(file)
        for key, value in raw_data.items():
            dataframe[key] = value
        return dataframe
    
    def get_data(self):
        return self._data
    
    def get_data_dict(self):
        data_dict = {}
        for key, _ in self._data[0].items():
            data_dict[key] = []
        data_length = len(self._data)
        i = 0
        for dataframe in self._data:
            i += 1
            print(f"Processing data: {i}/{data_length}", end='\r')
            for key, value in dataframe.items():
                data_dict[key].append(value)
        return data_dict
    
    def get_Hz(self):
        time_list = [dataframe['time'] for dataframe in self._data]
        time_diffs = [(time_list[i+1] - time_list[i]).total_seconds() for i in range(len(time_list)-1)]
        avg_time_diff = sum(time_diffs) / (len(time_diffs) + 1)
        Hz = 1 / avg_time_diff if avg_time_diff != 0 else 0
        return Hz
    
    def _time_order(self):
        self._data = sorted(self._data, key=lambda x:x['time'])

    def plot_time(self):
        data = self.get_data_dict()
        num = len(data) - 1
        time = data['time']
        start_time = time[0]
        for i in range(len(time)):
            time[i] = (time[i] - start_time).total_seconds()
        rows = 2
        cols = (num + 1) // rows
        fig, ax = plt.subplots(rows, cols, figsize=(15, 10))
        idx = 0
        for key, values in data.items():
            if key != 'time':
                r = idx // cols
                c = idx % cols

                ax[r, c].plot(time, values)
                ax[r, c].set_title(key)
                ax[r, c].legend([i for i in range(7)])
                idx += 1
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    data_processor = SJ_GelloDataProcessor(os.path.join(data_directory, target_data_directory))
    print(data_processor.get_Hz())
    data_processor.plot_time()