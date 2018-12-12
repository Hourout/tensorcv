import pandas as pd
import tensorflow as tf

class ImageClassificationFolderDataset():
    def __init__(self, root, image_format=['png', 'jpg', 'jpeg']):
        self.root = root
        self.image_format = image_format    

        file = tf.gfile.ListDirectory(self.root)
        file = [i for i in file if tf.gfile.IsDirectory(self.root+'/'+i) and i[0]!='.']
        data = pd.DataFrame()
        for i in file:
            data = pd.concat([data, pd.DataFrame({'image':tf.gfile.ListDirectory(self.root+'/'+i), 'label':i})])
        data = data.reset_index(drop=True)
        data['image'] = self.root+'/'+data.label+'/'+data.image
        data = data[data.image.map(lambda x: True if '.' in x.split('/')[-1] else False)]
        data = data[data.image.map(lambda x: True if x.split('/')[-1][0]!='.' else False)]
        data = data[data.image.map(lambda x: True if len(x.split('/')[-1].split('.'))==2 else False)]
        data = data[data.image.map(lambda x: True if str.lower(x.split('/')[-1].split('.')[1]) in self.image_format else False)]
        self.dataset = data.reset_index(drop=True)
        self.name_label_dict = {j: i for i, j in enumerate(data.label.unique())}
